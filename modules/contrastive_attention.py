"""
Contrastive Attention for Chest X-ray Report Generation

Implements the CA model from:
  Liu et al. "Contrastive Attention for Automatic Chest X-ray Report Generation"
  Findings of ACL-IJCNLP 2021, pages 269-280

The key idea: instead of only looking at the input image, COMPARE it with
known normal images to highlight what's DIFFERENT (i.e., abnormal).

Three steps:
  1. Build a normality pool P from training set normal images
  2. Aggregate Attention: find the closest normal images to the input
  3. Differentiate Attention: subtract common (normal) features from input,
     leaving contrastive (abnormal) features

This module operates on visual features AFTER the encoder, BEFORE the decoder.
It's orthogonal to the KG module — they can be combined.

Integration point in R2Gen:
  Original:  att_feats → Transformer Encoder → decoder input
  With CA:   att_feats → Transformer Encoder → ContrastiveAttention → decoder input

The CA-enhanced features emphasize abnormal regions, which helps the decoder
generate more accurate descriptions of pathology rather than defaulting to
generic "normal" phrases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os


class ContrastiveAttention(nn.Module):
    """
    Contrastive Attention (CA) model.

    Ref: Liu et al., ACL Findings 2021, Section 3.2

    Architecture:
      Input: visual features V [B, N_I, d] and global feature v_hat [B, d]
      Output: contrastive-enhanced features V_d [B, N_I, d]

      Step 1: Aggregate Attention finds closest normal images in pool
      Step 2: Differentiate Attention subtracts normal features from input
      Step 3: ReLU + Linear projects contrastive features

    The output V_d replaces V as input to the decoder, so the decoder
    attends to abnormality-highlighted features instead of raw visual features.
    """

    def __init__(self, d_model, pool_size=100, num_agg_rounds=3, dropout=0.1):
        """
        Args:
            d_model: Feature dimension (512 in R2Gen)
            pool_size: Number of normal images in the normality pool N_P.
                       Paper uses 1000 for MIMIC-CXR. We use 100 for IU X-Ray
                       (dataset is small, ~2770 training samples, ~70% normal).
                       Set higher for MIMIC-CXR (e.g., 500-1000).
            num_agg_rounds: Number of rounds n in Aggregate Attention.
                            Paper uses n=3 (Eq. 6). More rounds = better
                            filtering of noisy normal images.
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        self.num_agg_rounds = num_agg_rounds

        # Normality pool — registered as buffer so it moves to GPU with model
        # but is NOT a learnable parameter. Populated during build_normality_pool().
        # Shape: [N_P, d] where d = d_model
        self.register_buffer('normality_pool', torch.zeros(pool_size, d_model))
        self.pool_initialized = False

        # Aggregate Attention: learnable weights for n rounds
        # Eq. (6): alpha_k = softmax(W_k * v_hat) for k = 1..n
        # Each round uses a different projection to compute attention
        self.agg_projections = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(num_agg_rounds)
        ])

        # Differentiate Attention: projects common features
        # Eq. (8): v_hat' = Att(v_hat, V_agg)  where V_agg = closest normals
        # This is a standard attention: query=v_hat, key=value=V_agg
        self.diff_query_proj = nn.Linear(d_model, d_model)
        self.diff_key_proj = nn.Linear(d_model, d_model)
        self.diff_value_proj = nn.Linear(d_model, d_model)

        # Output projection: contrastive features
        # Eq. (9-10): v_d = ReLU(W_d * (v_hat - v_hat')) applied per-patch
        # We extend this to patch-level: subtract common from each patch
        self.contrast_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gating: control how much contrastive info to blend with original
        # This is our addition (not in the original paper) to ensure the
        # module doesn't hurt when there's no abnormality to contrast.
        # For normal images, the gate should be ~0 (keep original features).
        # For abnormal images, the gate should be >0 (inject contrastive info).
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, 1),
            nn.Sigmoid()
        )
        # Initialize gate bias negative so it starts conservative
        nn.init.constant_(self.gate[0].bias, -1.0)

        # Learnable scale for contrastive residual (ReZero-style)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # Init
        for proj in self.agg_projections:
            nn.init.xavier_uniform_(proj.weight)
        for m in [self.diff_query_proj, self.diff_key_proj, self.diff_value_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def build_normality_pool(self, visual_extractor, dataloader, dataset_name,
                             ann_path, device, max_pool_size=None):
        """
        Build the normality pool from training data.

        Called ONCE before training starts. Extracts global visual features
        from normal images in the training set and stores them.

        Ref: Liu et al. (2021), Section 3.2:
        "we first collect a normality pool P = {v_Normal_1, ..., v_Normal_NP}
         which consists of N_P normal images randomly extracted from the
         training dataset"

        How we identify "normal" images: reports containing keywords like
        "normal", "unremarkable", "clear", "no acute" and NOT containing
        abnormality keywords like "opacity", "effusion", "consolidation", etc.

        Args:
            visual_extractor: The ResNet visual extractor from R2Gen
            dataloader: Training dataloader
            dataset_name: 'iu_xray' or 'mimic_cxr'
            ann_path: Path to annotation.json
            device: torch device
            max_pool_size: Override pool size (None = use self.pool_size)
        """
        pool_size = max_pool_size or self.pool_size

        # Load annotations to identify normal reports
        ann = json.loads(open(ann_path, 'r').read())
        normal_ids = set()

        normal_keywords = {'normal', 'unremarkable', 'clear', 'no acute',
                           'within normal', 'intact', 'stable', 'negative'}
        abnormal_keywords = {'opacity', 'effusion', 'consolidation',
                             'atelectasis', 'pneumothorax', 'edema',
                             'cardiomegaly', 'infiltrate', 'mass', 'nodule',
                             'enlarged', 'fracture', 'thickening'}

        for example in ann['train']:
            report = example['report'].lower()
            has_normal = any(kw in report for kw in normal_keywords)
            has_abnormal = any(kw in report for kw in abnormal_keywords)
            if has_normal and not has_abnormal:
                normal_ids.add(example['id'])

        print(f"[CA] Found {len(normal_ids)} normal reports in training set")

        # Extract global visual features from normal images
        visual_extractor.eval()
        all_normal_feats = []

        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(dataloader):
                images = images.to(device)
                for i, img_id in enumerate(images_id):
                    if img_id in normal_ids and len(all_normal_feats) < pool_size * 2:
                        if dataset_name == 'iu_xray':
                            # IU X-Ray has 2 views — extract and concat
                            _, fc_0 = visual_extractor(images[i:i+1, 0])
                            _, fc_1 = visual_extractor(images[i:i+1, 1])
                            fc = torch.cat([fc_0, fc_1], dim=1)  # [1, d_vf*2]
                        else:
                            _, fc = visual_extractor(images[i:i+1])  # [1, d_vf]
                        all_normal_feats.append(fc.squeeze(0).cpu())

                if len(all_normal_feats) >= pool_size * 2:
                    break

        if len(all_normal_feats) == 0:
            print("[CA] WARNING: No normal images found! Using random pool.")
            self.normality_pool.normal_(0, 0.01)
            self.pool_initialized = True
            return

        # Stack and randomly sample pool_size features
        all_normal_feats = torch.stack(all_normal_feats)  # [K, d]
        if all_normal_feats.size(0) > pool_size:
            perm = torch.randperm(all_normal_feats.size(0))[:pool_size]
            all_normal_feats = all_normal_feats[perm]

        # The features from ResNet are d_vf (2048) or d_vf*2 (4096 for IU X-Ray)
        # but our module works in d_model (512). We need to project.
        # Store raw features; projection happens in forward via a linear layer.
        actual_dim = all_normal_feats.size(1)

        if actual_dim != self.d_model:
            # Add a projection layer
            self.pool_proj = nn.Linear(actual_dim, self.d_model, bias=False).to(device)
            nn.init.xavier_uniform_(self.pool_proj.weight)
            with torch.no_grad():
                projected = self.pool_proj(all_normal_feats.to(device))
                # Pad/truncate to pool_size
                actual_count = min(projected.size(0), pool_size)
                self.normality_pool[:actual_count] = projected[:actual_count]
        else:
            actual_count = min(all_normal_feats.size(0), pool_size)
            self.normality_pool[:actual_count] = all_normal_feats[:actual_count]

        self.pool_initialized = True
        print(f"[CA] Normality pool built: {actual_count} features, dim={self.d_model}")

    def _aggregate_attention(self, v_hat):
        """
        Aggregate Attention: find the closest normal images.

        Ref: Liu et al. (2021), Eq. (4-6)

        Performs n rounds of attention over the normality pool.
        Each round uses a different projection to compute attention weights,
        then averages the attended normal features.

        Args:
            v_hat: Global visual feature of input image [B, d]

        Returns:
            v_agg: Aggregated closest-normal features [B, d]
        """
        P = self.normality_pool  # [N_P, d]
        B = v_hat.size(0)
        N_P = P.size(0)

        # Accumulate attention-weighted pool across n rounds
        v_agg = torch.zeros_like(v_hat)  # [B, d]

        for k, proj in enumerate(self.agg_projections):
            # Eq. (5): attention scores between input and pool
            # score(v_hat, P_j) = (W_k * v_hat) · P_j / sqrt(d)
            q = proj(v_hat)  # [B, d]
            scores = torch.matmul(q, P.T) / (self.d_model ** 0.5)  # [B, N_P]
            weights = F.softmax(scores, dim=-1)  # [B, N_P]

            # Eq. (6): weighted sum of pool features
            attended = torch.matmul(weights, P)  # [B, d]
            v_agg = v_agg + attended

        # Average over rounds
        v_agg = v_agg / self.num_agg_rounds

        return v_agg

    def _differentiate_attention(self, v_hat, v_agg):
        """
        Differentiate Attention: extract what's different from normal.

        Ref: Liu et al. (2021), Eq. (7-10)

        Computes the "common" features between input and closest normals,
        then subtracts them to get contrastive features.

        Args:
            v_hat: Global visual feature of input [B, d]
            v_agg: Aggregated normal features [B, d]

        Returns:
            v_diff: Contrastive (difference) features [B, d]
        """
        # Eq. (8): Attention to find common features
        # v_hat' = attention(query=v_hat, key=v_agg, value=v_agg)
        # Since v_agg is already a single vector (not a sequence),
        # this simplifies to a projection-based comparison
        q = self.diff_query_proj(v_hat)     # [B, d]
        k = self.diff_key_proj(v_agg)       # [B, d]
        v = self.diff_value_proj(v_agg)     # [B, d]

        # Compute similarity and gate the common features
        sim = torch.sum(q * k, dim=-1, keepdim=True) / (self.d_model ** 0.5)
        attn_weight = torch.sigmoid(sim)  # [B, 1]

        v_common = attn_weight * v  # [B, d] — common (normal) features

        # Eq. (9): Subtract common from input to get contrastive
        v_diff = v_hat - v_common  # [B, d]

        # Eq. (10): Project through ReLU
        v_diff = self.contrast_proj(v_diff)  # [B, d]

        return v_diff

    def forward(self, att_feats, fc_feats):
        """
        Apply Contrastive Attention to enhance visual features.

        Args:
            att_feats: Patch-level visual features [B, N_I, d_model]
                       (already projected from d_vf to d_model by att_embed)
            fc_feats: Global (pooled) visual features [B, d_visual]
                      (raw from ResNet, NOT projected yet)

        Returns:
            enhanced_feats: Contrastive-enhanced patch features [B, N_I, d_model]
        """
        B, N_I, D = att_feats.shape

        if not self.pool_initialized:
            # Pool not built yet — pass through unchanged
            return att_feats

        # Project fc_feats to d_model if needed
        if fc_feats.size(-1) != self.d_model:
            if hasattr(self, 'pool_proj'):
                v_hat = self.pool_proj(fc_feats)  # [B, d_model]
            else:
                # Fallback: average pool the att_feats
                v_hat = att_feats.mean(dim=1)  # [B, d_model]
        else:
            v_hat = fc_feats

        # Step 1: Aggregate Attention — find closest normals
        v_agg = self._aggregate_attention(v_hat)  # [B, d]

        # Step 2: Differentiate Attention — extract abnormality signal
        v_diff = self._differentiate_attention(v_hat, v_agg)  # [B, d]

        # Step 3: Apply contrastive features to patch-level attention features
        # Expand v_diff from [B, d] to [B, N_I, d] and add to each patch
        v_diff_expanded = v_diff.unsqueeze(1).expand_as(att_feats)  # [B, N_I, d]

        # Gated residual: let the model learn when to use contrastive info
        gate_input = torch.cat([att_feats, v_diff_expanded], dim=-1)  # [B, N_I, 2d]
        g = self.gate(gate_input)  # [B, N_I, 1]

        enhanced = att_feats + self.residual_scale * g * v_diff_expanded

        return enhanced