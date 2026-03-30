"""
Contrastive Attention for Chest X-ray Report Generation
Ref: Liu et al. (ACL Findings 2021)

Operates on visual features AFTER att_embed, BEFORE Transformer encoder.
Orthogonal to KG module — they combine independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class ContrastiveAttention(nn.Module):
    def __init__(self, d_model, pool_size=100, num_agg_rounds=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        self.num_agg_rounds = num_agg_rounds
        self.register_buffer('normality_pool', torch.zeros(pool_size, d_model))
        self.pool_initialized = False

        self.agg_projections = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_agg_rounds)
        ])
        self.diff_query_proj = nn.Linear(d_model, d_model)
        self.diff_key_proj = nn.Linear(d_model, d_model)
        self.diff_value_proj = nn.Linear(d_model, d_model)
        self.contrast_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(d_model * 2, 1), nn.Sigmoid())
        nn.init.constant_(self.gate[0].bias, -1.0)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        for proj in self.agg_projections:
            nn.init.xavier_uniform_(proj.weight)
        for m in [self.diff_query_proj, self.diff_key_proj, self.diff_value_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def build_normality_pool(self, visual_extractor, dataloader, dataset_name,
                             ann_path, device, max_pool_size=None):
        pool_size = max_pool_size or self.pool_size
        ann = json.loads(open(ann_path, 'r').read())
        normal_ids = set()
        normal_kw = {'normal', 'unremarkable', 'clear', 'no acute',
                     'within normal', 'intact', 'stable', 'negative'}
        abnormal_kw = {'opacity', 'effusion', 'consolidation', 'atelectasis',
                       'pneumothorax', 'edema', 'cardiomegaly', 'infiltrate',
                       'mass', 'nodule', 'enlarged', 'fracture', 'thickening'}
        for ex in ann['train']:
            report = ex['report'].lower()
            if any(k in report for k in normal_kw) and not any(k in report for k in abnormal_kw):
                normal_ids.add(ex['id'])
        print(f"[CA] Found {len(normal_ids)} normal reports")

        visual_extractor.eval()
        all_feats = []
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(dataloader):
                images = images.to(device)
                for i, img_id in enumerate(images_id):
                    if img_id in normal_ids and len(all_feats) < pool_size * 2:
                        if dataset_name == 'iu_xray':
                            _, fc_0 = visual_extractor(images[i:i+1, 0])
                            _, fc_1 = visual_extractor(images[i:i+1, 1])
                            fc = torch.cat([fc_0, fc_1], dim=1)
                        else:
                            _, fc = visual_extractor(images[i:i+1])
                        all_feats.append(fc.squeeze(0).cpu())
                if len(all_feats) >= pool_size * 2:
                    break

        if len(all_feats) == 0:
            print("[CA] WARNING: No normal images found! Using random pool.")
            self.normality_pool.normal_(0, 0.01)
            self.pool_initialized = True
            return

        all_feats = torch.stack(all_feats)
        if all_feats.size(0) > pool_size:
            perm = torch.randperm(all_feats.size(0))[:pool_size]
            all_feats = all_feats[perm]

        actual_dim = all_feats.size(1)
        if actual_dim != self.d_model:
            self.pool_proj = nn.Linear(actual_dim, self.d_model, bias=False).to(device)
            nn.init.xavier_uniform_(self.pool_proj.weight)
            with torch.no_grad():
                projected = self.pool_proj(all_feats.to(device))
                n = min(projected.size(0), pool_size)
                self.normality_pool[:n] = projected[:n]
        else:
            n = min(all_feats.size(0), pool_size)
            self.normality_pool[:n] = all_feats[:n]

        self.pool_initialized = True
        print(f"[CA] Normality pool built: {n} features, dim={self.d_model}")

    def _aggregate_attention(self, v_hat):
        P = self.normality_pool
        v_agg = torch.zeros_like(v_hat)
        for proj in self.agg_projections:
            q = proj(v_hat)
            scores = torch.matmul(q, P.T) / (self.d_model ** 0.5)
            weights = F.softmax(scores, dim=-1)
            v_agg = v_agg + torch.matmul(weights, P)
        return v_agg / self.num_agg_rounds

    def _differentiate_attention(self, v_hat, v_agg):
        q = self.diff_query_proj(v_hat)
        k = self.diff_key_proj(v_agg)
        v = self.diff_value_proj(v_agg)
        sim = torch.sum(q * k, dim=-1, keepdim=True) / (self.d_model ** 0.5)
        v_common = torch.sigmoid(sim) * v
        return self.contrast_proj(v_hat - v_common)

    def forward(self, att_feats, fc_feats):
        B, N_I, D = att_feats.shape
        if not self.pool_initialized:
            return att_feats

        if fc_feats.size(-1) != self.d_model:
            if hasattr(self, 'pool_proj'):
                v_hat = self.pool_proj(fc_feats)
            else:
                v_hat = att_feats.mean(dim=1)
        else:
            v_hat = fc_feats

        v_agg = self._aggregate_attention(v_hat)
        v_diff = self._differentiate_attention(v_hat, v_agg)
        v_diff_exp = v_diff.unsqueeze(1).expand_as(att_feats)
        g = self.gate(torch.cat([att_feats, v_diff_exp], dim=-1))
        return att_feats + self.residual_scale * g * v_diff_exp