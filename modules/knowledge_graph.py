"""
Knowledge Graph Module for R2Gen (v2 — fixed)

Fixes from v1:
- FIX A: Image-conditioned KG node activation via visual features
  The GCN output is now GATED per-image using a visual-to-graph projection.
  Ref: Zhang et al. (AAAI 2020) — classifier output gates KG nodes
  Ref: PPKED (CVPR 2021) — PrKE uses visual features to query the KG

- FIX B: Anti-over-smoothing in GCN
  Reduced to 1 GCN layer with stronger residual (alpha-weighted).
  Ref: Li et al. "Deeper Insights into GCN for Semi-Supervised Learning" (AAAI 2018)

- FIX C: Removed KGVocabularyBias (was reinforcing mode collapse)

- FIX D: Lightweight cross-attention with fewer new parameters
  Reduced heads, scalar gate, ReZero-style residual scaling.

Paper references:
- Zhang et al. "When Radiology Report Generation Meets KG" (AAAI 2020)
- Kipf & Welling "Semi-Supervised Classification with GCNs" (ICLR 2017)
- KiUT, Huang et al. (CVPR 2023)
- DCG, Liang et al. (ACM MM 2024)
- PPKED, Liu et al. (CVPR 2021)
"""

import json
import re
import math
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Entity Definitions
# =============================================================================

ANATOMY_ENTITIES = [
    'lung', 'lungs', 'heart', 'cardiac', 'mediastinum', 'mediastinal',
    'pleural', 'aorta', 'aortic', 'thoracic', 'diaphragm', 'rib', 'ribs',
    'spine', 'vertebral', 'costophrenic', 'hilum', 'hilar', 'trachea',
    'bronchial', 'pulmonary', 'chest', 'sternum', 'clavicle', 'shoulder',
    'abdomen', 'airspace', 'lobe', 'apex', 'base', 'vasculature',
    'vascular', 'silhouette', 'bony', 'osseous', 'skeletal',
]

FINDING_ENTITIES = {
    'abnormal': [
        'opacity', 'opacities', 'effusion', 'effusions', 'consolidation',
        'atelectasis', 'pneumothorax', 'edema', 'cardiomegaly', 'infiltrate',
        'infiltrates', 'nodule', 'nodules', 'mass', 'lesion', 'fracture',
        'calcification', 'thickening', 'congestion', 'fibrosis',
        'pneumonia', 'emphysema', 'hernia', 'granuloma', 'scoliosis',
        'kyphosis', 'widening', 'tortuous', 'tortuosity', 'prominent',
        'enlarged', 'enlargement', 'elevated', 'flattening', 'blunting',
        'scarring', 'deformity', 'degenerative', 'hyperinflation',
        'hyperinflated', 'hypoinflation',
    ],
    'normal': [
        'normal', 'clear', 'unremarkable', 'stable', 'intact', 'midline',
        'symmetric', 'preserved', 'adequate', 'appropriate', 'satisfactory',
        'no acute', 'within normal', 'negative', 'free',
    ]
}

SYNONYM_MAP = {
    'lungs': 'lung', 'cardiac': 'heart', 'mediastinal': 'mediastinum',
    'aortic': 'aorta', 'hilar': 'hilum', 'ribs': 'rib',
    'opacities': 'opacity', 'effusions': 'effusion',
    'infiltrates': 'infiltrate', 'nodules': 'nodule',
    'enlarged': 'enlargement', 'tortuous': 'tortuosity',
    'hyperinflated': 'hyperinflation',
}


# =============================================================================
# 2. Knowledge Graph Construction
# =============================================================================

class KnowledgeGraphBuilder:
    """Builds medical KG from report annotations."""
    
    def __init__(self, ann_path, dataset_name='iu_xray', co_occur_threshold=3):
        self.ann_path = ann_path
        self.dataset_name = dataset_name
        self.co_occur_threshold = co_occur_threshold
        self.all_finding_words = FINDING_ENTITIES['abnormal'] + FINDING_ENTITIES['normal']
        self.all_entity_words = ANATOMY_ENTITIES + self.all_finding_words
        
    def _clean_report(self, report):
        report = report.replace('..', '.').replace('..', '.').strip().lower()
        report = re.sub('[.,?;*!%^&_+():\\-\\[\\]{}]', ' ', report)
        report = re.sub('\\s+', ' ', report)
        return report
    
    def _extract_entities(self, report_text):
        clean = self._clean_report(report_text)
        words = clean.split()
        found = set()
        for word in words:
            canonical = SYNONYM_MAP.get(word, word)
            if canonical in self.all_entity_words:
                found.add(canonical)
        for i in range(len(words) - 1):
            bigram = words[i] + ' ' + words[i+1]
            if bigram in self.all_finding_words:
                found.add(bigram)
        return found
    
    def _get_node_type(self, entity):
        if entity in ANATOMY_ENTITIES:
            return 'anatomy'
        elif entity in FINDING_ENTITIES['abnormal']:
            return 'abnormal'
        elif entity in FINDING_ENTITIES['normal']:
            return 'normal'
        return 'anatomy'
    
    def build(self, split='train'):
        ann = json.loads(open(self.ann_path, 'r').read())
        reports = ann[split]
        
        entity_counter = Counter()
        co_occurrence = defaultdict(int)
        for example in reports:
            entities = self._extract_entities(example['report'])
            for e in entities:
                entity_counter[e] += 1
            for e1, e2 in combinations(sorted(entities), 2):
                co_occurrence[(e1, e2)] += 1
                
        min_freq = 2
        valid = {e for e, c in entity_counter.items() if c >= min_freq}
        
        anat = sorted([e for e in valid if self._get_node_type(e) == 'anatomy'])
        abnl = sorted([e for e in valid if self._get_node_type(e) == 'abnormal'])
        norm = sorted([e for e in valid if self._get_node_type(e) == 'normal'])
        
        node_list = anat + abnl + norm
        node_types = ['anatomy']*len(anat) + ['abnormal']*len(abnl) + ['normal']*len(norm)
        node2idx = {name: idx for idx, name in enumerate(node_list)}
        
        N = len(node_list)
        adj = np.zeros((N, N), dtype=np.float32)
        for (e1, e2), count in co_occurrence.items():
            if e1 in node2idx and e2 in node2idx and count >= self.co_occur_threshold:
                i, j = node2idx[e1], node2idx[e2]
                adj[i, j] = count
                adj[j, i] = count
        
        adj += np.eye(N)
        deg = adj.sum(axis=1)
        d_inv = np.power(deg, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        adj = np.diag(d_inv) @ adj @ np.diag(d_inv)
        
        print(f"[KG] {N} nodes ({len(anat)} anat, {len(abnl)} abnl, {len(norm)} norm)")
        return node_list, node_types, adj, node2idx
    
    def extract_labels_for_report(self, report_text, node_list, node2idx):
        entities = self._extract_entities(report_text)
        labels = np.zeros(len(node_list), dtype=np.float32)
        for e in entities:
            if e in node2idx:
                labels[node2idx[e]] = 1.0
        return labels


# =============================================================================
# 3. GCN Layer
# =============================================================================

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        stdv = 1. / math.sqrt(out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        out = torch.matmul(adj, support) if x.dim() == 2 else torch.matmul(adj.unsqueeze(0), support)
        return out + self.bias if self.bias is not None else out


# =============================================================================
# 4. KG Encoder — FIXED
# =============================================================================

class KnowledgeGraphEncoder(nn.Module):
    """
    GCN encoder with image-conditioned node gating.
    
    FIX A: visual_to_node_gate produces per-image, per-node activation.
    FIX B: alpha-weighted residual prevents over-smoothing.
    """
    
    def __init__(self, num_nodes, node_types, d_model, d_visual,
                 num_gcn_layers=1, dropout=0.1, gcn_residual_alpha=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.alpha = gcn_residual_alpha
        
        self.node_embeddings = nn.Embedding(num_nodes, d_model)
        self.type_map = {'anatomy': 0, 'abnormal': 1, 'normal': 2}
        self.register_buffer('type_ids', torch.LongTensor([self.type_map[t] for t in node_types]))
        self.type_embeddings = nn.Embedding(3, d_model)
        
        self.gcn_layers = nn.ModuleList([GraphConvolution(d_model, d_model) for _ in range(num_gcn_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # FIX A: Image → per-node gate
        self.visual_to_node_gate = nn.Sequential(
            nn.Linear(d_visual, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_nodes),
        )
        
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        nn.init.xavier_uniform_(self.type_embeddings.weight)
        
    def forward(self, adj, fc_feats):
        """
        Args:
            adj: [N, N] normalized adjacency
            fc_feats: [B, d_visual] pooled visual features
        Returns:
            kg_feats: [B, N, d_model] image-conditioned
        """
        B = fc_feats.size(0)
        node_ids = torch.arange(self.num_nodes, device=adj.device)
        x = self.node_embeddings(node_ids) + self.type_embeddings(self.type_ids)
        
        for gcn in self.gcn_layers:
            residual = x
            out = F.relu(self.dropout(gcn(x, adj)))
            x = self.alpha * out + (1 - self.alpha) * residual
        x = self.norm(x)  # [N, D]
        
        # Per-image gating
        gates = torch.sigmoid(self.visual_to_node_gate(fc_feats))  # [B, N]
        kg_feats = x.unsqueeze(0).expand(B, -1, -1) * gates.unsqueeze(-1)  # [B, N, D]
        return kg_feats


# =============================================================================
# 5. Multi-Label Classifier
# =============================================================================

class KGMultiLabelClassifier(nn.Module):
    def __init__(self, visual_feat_dim, num_nodes, d_model=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(visual_feat_dim, d_model), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_model, num_nodes),
        )
    def forward(self, visual_feats):
        return self.projection(visual_feats)
    def get_loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)


# =============================================================================
# 6. KG Cross-Attention — FIXED
# =============================================================================

class KGCrossAttention(nn.Module):
    """
    Lightweight cross-attention. KG feats are now [B,N,D] (per-image).
    Scalar gate + ReZero-style residual scaling.
    """
    
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.gate_linear = nn.Linear(d_model * 2, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
        nn.init.constant_(self.gate_linear.bias, -2.0)
    
    def forward(self, decoder_hidden, kg_feats):
        """
        Args:
            decoder_hidden: [B, L, D]
            kg_feats: [B, N, D] (image-conditioned)
        Returns:
            output: [B, L, D]
        """
        B, L, D = decoder_hidden.shape
        residual = decoder_hidden
        x = self.norm(decoder_hidden)
        N = kg_feats.size(1)
        
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(kg_feats).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(kg_feats).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, D)
        ctx = self.W_o(ctx)
        
        g = torch.sigmoid(self.gate_linear(torch.cat([residual, ctx], dim=-1)))
        return residual + self.residual_scale * g * self.dropout(ctx)


# =============================================================================
# 7. KG Alignment Loss
# =============================================================================

class KGAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, kg_attention_weights, kg_labels):
        avg_attn = kg_attention_weights.mean(dim=1)
        return F.binary_cross_entropy(avg_attn.clamp(1e-7, 1-1e-7), kg_labels)