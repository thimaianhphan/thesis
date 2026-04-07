"""
Contrastive Attention for Chest X-ray Report Generation
Ref: Liu et al. (ACL Findings 2021)

Changes in v3:
  - build_normality_pool() now accepts a KnowledgeGraphBuilder and uses
    kg_builder.is_normal_report() (BiomedCLIP-based) instead of keyword matching.
  - fc_proj is always defined in __init__ — no conditional pool_proj.
  - forward() always uses fc_proj(fc_feats) — no fallback.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAttention(nn.Module):
    def __init__(self, d_model, d_fc, pool_size=100, num_agg_rounds=3, dropout=0.1):
        """
        Args:
            d_model   : Transformer hidden dim (512)
            d_fc      : Raw visual extractor fc dim (256 for MedSAM, 2048/4096 for ResNet)
            pool_size : Number of normal images to keep in pool
            num_agg_rounds : Rounds n in Aggregate Attention (paper: 3)
        """
        super().__init__()
        self.d_model       = d_model
        self.d_fc          = d_fc
        self.pool_size     = pool_size
        self.num_agg_rounds = num_agg_rounds
        self.register_buffer('normality_pool', torch.zeros(pool_size, d_model))
        self.pool_initialized = False

        # Always-present projection: fc → d_model
        self.fc_proj = nn.Linear(d_fc, d_model, bias=False)
        nn.init.xavier_uniform_(self.fc_proj.weight)

        self.agg_projections = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_agg_rounds)
        ])
        self.diff_query_proj = nn.Linear(d_model, d_model)
        self.diff_key_proj   = nn.Linear(d_model, d_model)
        self.diff_value_proj = nn.Linear(d_model, d_model)
        self.contrast_proj   = nn.Sequential(
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

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def build_normality_pool(self, visual_extractor, dataloader, dataset_name,
                             ann_path, device, max_pool_size=None,
                             kg_builder=None):
        """
        Build the normality pool from training images whose reports are
        classified as normal by BiomedCLIP (via kg_builder.is_normal_report).

        Falls back to keyword matching if kg_builder is not provided.

        Args:
            visual_extractor : the model's visual extractor (frozen during pool build)
            dataloader       : training DataLoader
            dataset_name     : 'iu_xray' or 'mimic_cxr'
            ann_path         : path to annotation JSON
            device           : torch device
            max_pool_size    : override self.pool_size if given
            kg_builder       : KnowledgeGraphBuilder (has is_normal_report())
        """
        pool_size = max_pool_size or self.pool_size
        ann = json.loads(open(ann_path, 'r').read())

        # Build set of normal image IDs
        normal_ids = set()
        print("[CA] Identifying normal reports...")

        if kg_builder is not None:
            # BiomedCLIP-based: accurate, domain-appropriate
            for ex in ann['train']:
                if kg_builder.is_normal_report(ex['report']):
                    normal_ids.add(ex['id'])
            print(f"[CA] BiomedCLIP found {len(normal_ids)} normal reports")
        else:
            # Fallback: keyword heuristic (fast but imprecise)
            normal_kw   = {'normal', 'unremarkable', 'clear', 'no acute',
                           'within normal', 'intact', 'stable', 'negative'}
            abnormal_kw = {'opacity', 'effusion', 'consolidation', 'atelectasis',
                           'pneumothorax', 'edema', 'cardiomegaly', 'infiltrate',
                           'mass', 'nodule', 'enlarged', 'fracture', 'thickening'}
            for ex in ann['train']:
                report = ex['report'].lower()
                if (any(k in report for k in normal_kw) and
                        not any(k in report for k in abnormal_kw)):
                    normal_ids.add(ex['id'])
            print(f"[CA] Keyword heuristic found {len(normal_ids)} normal reports")

        # Collect fc features from normal images
        visual_extractor.eval()
        all_feats = []
        with torch.no_grad():
            for images_id, images, reports_ids, reports_masks in dataloader:
                images = images.to(device)
                for i, img_id in enumerate(images_id):
                    if img_id not in normal_ids:
                        continue
                    if len(all_feats) >= pool_size * 2:
                        break
                    if dataset_name == 'iu_xray':
                        _, fc_0 = visual_extractor(images[i:i+1, 0])
                        _, fc_1 = visual_extractor(images[i:i+1, 1])
                        fc = torch.cat([fc_0, fc_1], dim=1)
                    else:
                        _, fc = visual_extractor(images[i:i+1])
                    # Project to d_model using our fc_proj
                    projected = self.fc_proj(fc.to(device))   # [1, d_model]
                    all_feats.append(projected.squeeze(0).cpu())
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

        n = all_feats.size(0)
        self.normality_pool[:n] = all_feats.to(self.normality_pool.device)
        self.pool_initialized = True
        print(f"[CA] Normality pool built: {n} features, dim={self.d_model}")

    # ------------------------------------------------------------------
    # Core CA operations
    # ------------------------------------------------------------------

    def _aggregate_attention(self, v_hat):
        P = self.normality_pool                              # [pool_size, d_model]
        v_agg = torch.zeros_like(v_hat)
        for proj in self.agg_projections:
            q      = proj(v_hat)                            # [B, d_model]
            scores = torch.matmul(q, P.T) / (self.d_model ** 0.5)  # [B, pool_size]
            weights = F.softmax(scores, dim=-1)
            v_agg  = v_agg + torch.matmul(weights, P)      # [B, d_model]
        return v_agg / self.num_agg_rounds

    def _differentiate_attention(self, v_hat, v_agg):
        q = self.diff_query_proj(v_hat)
        k = self.diff_key_proj(v_agg)
        v = self.diff_value_proj(v_agg)
        sim      = torch.sum(q * k, dim=-1, keepdim=True) / (self.d_model ** 0.5)
        v_common = torch.sigmoid(sim) * v
        return self.contrast_proj(v_hat - v_common)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, att_feats, fc_feats):
        """
        Args:
            att_feats : [B, N_patches, d_model]  — post-att_embed patch features
            fc_feats  : [B, d_fc]                — raw visual extractor fc output

        Returns:
            att_feats : [B, N_patches, d_model]  — CA-enhanced
        """
        if not self.pool_initialized:
            return att_feats

        v_hat  = self.fc_proj(fc_feats)                         # [B, d_model]
        v_agg  = self._aggregate_attention(v_hat)               # [B, d_model]
        v_diff = self._differentiate_attention(v_hat, v_agg)   # [B, d_model]

        v_diff_exp = v_diff.unsqueeze(1).expand_as(att_feats)  # [B, N, d_model]
        g = self.gate(torch.cat([att_feats, v_diff_exp], dim=-1))  # [B, N, 1]

        return att_feats + self.residual_scale * g * v_diff_exp