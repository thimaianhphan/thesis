"""
Knowledge Graph Module for R2Gen — v3: BiomedCLIP corpus-based node discovery

Changes from v2:
  - ANATOMY_ENTITIES / FINDING_ENTITIES hardcoded lists are GONE.
  - KnowledgeGraphBuilder.build() now runs in two phases:
      Phase 1 (corpus scan): extract all unigrams/bigrams that appear >= min_freq
                             times across training reports. Cheap, no model needed.
      Phase 2 (BiomedCLIP semantic typing): embed every candidate term with the
                             BiomedCLIP text encoder, then cluster into
                             anatomy / abnormal / normal via cosine similarity
                             to three anchor prompts. This replaces the hardcoded
                             word lists entirely.
  - ContrastiveAttention normality pool now also uses BiomedCLIP anchors to
    decide which training images are "normal" instead of keyword matching.

Architecture is unchanged: KG fuses into the DECODER via cross-attention.

Paper references:
  - Zhang et al. "When Radiology Report Generation Meets KG" (AAAI 2020)
  - Kipf & Welling "Semi-Supervised Classification with GCNs" (ICLR 2017)
  - KiUT, Huang et al. (CVPR 2023)
  - DCG, Liang et al. (ACM MM 2024)  ← BiomedCLIP semantic typing idea
  - BiomedCLIP, Zhang et al. (2023)
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
# 1. BiomedCLIP Text Encoder — lazy-loaded, frozen, CPU/GPU agnostic
# =============================================================================

_BIOMEDCLIP_MODEL = None
_BIOMEDCLIP_TOKENIZER = None
_BIOMEDCLIP_DEVICE = None

def _load_biomedclip(device='cpu'):
    """Load BiomedCLIP once and cache it globally. Frozen — no grad."""
    global _BIOMEDCLIP_MODEL, _BIOMEDCLIP_TOKENIZER, _BIOMEDCLIP_DEVICE
    if _BIOMEDCLIP_MODEL is None:
        try:
            from open_clip import create_model_from_pretrained, get_tokenizer
        except ImportError:
            raise ImportError(
                "open_clip is required for BiomedCLIP. "
                "Install with: pip install open_clip_torch"
            )
        print("[KG] Loading BiomedCLIP text encoder (one-time)...")
        model, _ = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        _BIOMEDCLIP_MODEL = model
        _BIOMEDCLIP_TOKENIZER = tokenizer
        _BIOMEDCLIP_DEVICE = device
        print("[KG] BiomedCLIP loaded.")
    return _BIOMEDCLIP_MODEL, _BIOMEDCLIP_TOKENIZER


@torch.no_grad()
def biomedclip_encode_text(phrases, device='cpu', batch_size=64):
    """
    Encode a list of phrases with BiomedCLIP's text encoder.
    Returns L2-normalised embeddings [N, 512].
    """
    model, tokenizer = _load_biomedclip(device)
    all_embs = []
    for i in range(0, len(phrases), batch_size):
        batch = phrases[i:i + batch_size]
        # BiomedCLIP template from the official model card
        prompts = [f"a radiology image showing {p}" for p in batch]
        tokens = tokenizer(prompts).to(device)
        embs = model.encode_text(tokens)                  # [B, 512]
        embs = F.normalize(embs.float(), dim=-1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)                     # [N, 512]


# Anchor phrases that define the three semantic clusters
_ANCHOR_PHRASES = {
    'anatomy':  "chest anatomy structure lung heart mediastinum",
    'abnormal': "abnormal radiological finding disease pathology opacity effusion",
    'normal':   "normal clear unremarkable no acute finding stable",
}

def classify_terms_with_biomedclip(terms, device='cpu', threshold=0.0):
    """
    Classify each term in `terms` into one of {anatomy, abnormal, normal}
    by comparing its BiomedCLIP embedding to three anchor embeddings.

    Returns: dict mapping term -> type_str
    """
    if not terms:
        return {}

    anchor_texts = [_ANCHOR_PHRASES[k] for k in ('anatomy', 'abnormal', 'normal')]
    anchor_embs = biomedclip_encode_text(anchor_texts, device=device)  # [3, 512]
    term_embs = biomedclip_encode_text(list(terms), device=device)      # [N, 512]

    # cosine similarity: already L2-normalised → just dot product
    sims = torch.matmul(term_embs, anchor_embs.T)  # [N, 3]
    best = sims.argmax(dim=-1).tolist()
    type_names = ['anatomy', 'abnormal', 'normal']

    result = {}
    for term, idx in zip(terms, best):
        result[term] = type_names[idx]
    return result


# =============================================================================
# 2. Knowledge Graph Construction — corpus-driven, BiomedCLIP-typed
# =============================================================================

# Stopwords to skip during candidate extraction
_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'of', 'in', 'on', 'at',
    'to', 'for', 'with', 'by', 'from', 'as', 'into', 'through', 'during',
    'this', 'that', 'these', 'those', 'and', 'or', 'but', 'not', 'no',
    'also', 'both', 'each', 'more', 'most', 'other', 'some', 'such',
    'than', 'too', 'very', 'just', 'then', 'there', 'when', 'where',
    'which', 'who', 'whom', 'how', 'what', 'if', 'so', 'up', 'out',
    'about', 'above', 'after', 'again', 'against', 'all', 'any',
    'because', 'before', 'between', 'further', 'here', 'its', 'only',
    'our', 'own', 'same', 'their', 'them', 'they', 'we', 'your',
    'without', 'noted', 'seen', 'note', 'patient', 'exam', 'image',
    'images', 'view', 'views', 'radiograph', 'radiographs', 'chest',
    'comparison', 'unchanged', 'change', 'new', 'old', 'prior', 'now',
    'interval', 'since', 'previously', 'however', 'well', 'including',
}


class KnowledgeGraphBuilder:
    def __init__(self, ann_path, dataset_name='iu_xray',
                 co_occur_threshold=3,
                 min_term_freq=5,
                 max_nodes=150,
                 biomedclip_device='cpu'):
        """
        Args:
            ann_path:             Path to annotations JSON
            dataset_name:         'iu_xray' or 'mimic_cxr'
            co_occur_threshold:   Min co-occurrence count for an edge
            min_term_freq:        Min document frequency for a term to be a node
            max_nodes:            Hard cap on total nodes (keeps graph small)
            biomedclip_device:    Device for BiomedCLIP ('cpu' or 'cuda')
        """
        self.ann_path = ann_path
        self.dataset_name = dataset_name
        self.co_occur_threshold = co_occur_threshold
        self.min_term_freq = min_term_freq
        self.max_nodes = max_nodes
        self.biomedclip_device = biomedclip_device

        # Populated by build()
        self._term_embeddings = None   # {term: tensor [512]}  — for label extraction
        self._node_list = None
        self._node2idx = None

    # ------------------------------------------------------------------
    # Internal: report cleaning
    # ------------------------------------------------------------------

    def _clean_report(self, report):
        report = report.lower().strip()
        report = report.replace('..', '.').replace('..', '.')
        report = re.sub(r'[.,?;*!%^&_+():\-\[\]{}]', ' ', report)
        report = re.sub(r'\s+', ' ', report)
        return report

    def _extract_candidates(self, report_text):
        """Extract unigram and bigram candidates from a single report."""
        clean = self._clean_report(report_text)
        words = [w for w in clean.split() if w not in _STOPWORDS and len(w) > 2]
        candidates = set(words)
        for i in range(len(words) - 1):
            bigram = words[i] + ' ' + words[i + 1]
            candidates.add(bigram)
        return candidates

    # ------------------------------------------------------------------
    # Public: build graph
    # ------------------------------------------------------------------

    def build(self, split='train'):
        """
        Build the knowledge graph from the training corpus.

        Returns:
            node_list  : list of str, node names
            node_types : list of str, one of {anatomy, abnormal, normal}
            adjacency  : np.ndarray [N, N], normalised adjacency matrix
            node2idx   : dict {name: int}
        """
        ann = json.loads(open(self.ann_path, 'r').read())
        reports = ann[split]

        # ---- Phase 1: corpus frequency scan ----
        term_doc_freq = Counter()
        co_occurrence = defaultdict(int)

        print(f"[KG] Phase 1: scanning {len(reports)} reports for candidate terms...")
        for example in reports:
            candidates = self._extract_candidates(example['report'])
            for t in candidates:
                term_doc_freq[t] += 1
            for t1, t2 in combinations(sorted(candidates), 2):
                co_occurrence[(t1, t2)] += 1

        # Filter by frequency
        valid_terms = sorted([
            t for t, f in term_doc_freq.items()
            if f >= self.min_term_freq
        ], key=lambda t: -term_doc_freq[t])

        # Cap at max_nodes * 3 before BiomedCLIP (we'll cap again after typing)
        if len(valid_terms) > self.max_nodes * 3:
            valid_terms = valid_terms[:self.max_nodes * 3]

        print(f"[KG] Phase 1 complete: {len(valid_terms)} candidates "
              f"(freq >= {self.min_term_freq})")

        # ---- Phase 2: BiomedCLIP semantic typing ----
        print(f"[KG] Phase 2: BiomedCLIP typing of {len(valid_terms)} terms...")
        type_map = classify_terms_with_biomedclip(
            valid_terms, device=self.biomedclip_device
        )
        print(f"[KG] Typing complete.")

        # Group by type, sort by frequency descending, cap per type
        per_type = defaultdict(list)
        for term in valid_terms:
            per_type[type_map[term]].append(term)

        # Balance: anatomy gets 40%, abnormal 40%, normal 20%
        n_anat  = int(self.max_nodes * 0.40)
        n_abnl  = int(self.max_nodes * 0.40)
        n_norm  = self.max_nodes - n_anat - n_abnl

        anat = per_type['anatomy'][:n_anat]
        abnl = per_type['abnormal'][:n_abnl]
        norm = per_type['normal'][:n_norm]

        node_list  = anat + abnl + norm
        node_types = (['anatomy'] * len(anat) +
                      ['abnormal'] * len(abnl) +
                      ['normal'] * len(norm))
        node2idx   = {name: idx for idx, name in enumerate(node_list)}
        N = len(node_list)

        # ---- Build adjacency matrix ----
        adj = np.zeros((N, N), dtype=np.float32)
        for (t1, t2), count in co_occurrence.items():
            if t1 in node2idx and t2 in node2idx and count >= self.co_occur_threshold:
                i, j = node2idx[t1], node2idx[t2]
                adj[i, j] = count
                adj[j, i] = count

        # Symmetrically normalise: D^{-1/2} A D^{-1/2}
        adj += np.eye(N)
        deg = adj.sum(axis=1)
        d_inv = np.power(deg, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        adj = np.diag(d_inv) @ adj @ np.diag(d_inv)

        # Cache for label extraction
        self._node_list = node_list
        self._node2idx  = node2idx

        print(f"[KG] Graph built: {N} nodes "
              f"({len(anat)} anatomy, {len(abnl)} abnormal, {len(norm)} normal)")
        print(f"[KG] Sample anatomy : {anat[:5]}")
        print(f"[KG] Sample abnormal: {abnl[:5]}")
        print(f"[KG] Sample normal  : {norm[:5]}")

        return node_list, node_types, adj, node2idx

    # ------------------------------------------------------------------
    # Label extraction for KG alignment loss (used during training)
    # ------------------------------------------------------------------

    def extract_labels_for_report(self, report_text, node_list, node2idx):
        """
        Produce a binary label vector [N] indicating which KG nodes
        appear (as a substring/word match) in this report.
        BiomedCLIP-based: we embed the report and score against node embeddings.
        Falls back to substring matching for speed during training.
        """
        clean = self._clean_report(report_text)
        labels = np.zeros(len(node_list), dtype=np.float32)
        for i, node in enumerate(node_list):
            # Use word-boundary aware match so 'lung' doesn't match 'lungs' etc.
            # Simple regex: full word or phrase match after cleaning
            pattern = r'\b' + re.escape(node) + r'\b'
            if re.search(pattern, clean):
                labels[i] = 1.0
        return labels

    # ------------------------------------------------------------------
    # BiomedCLIP-based normality scoring (used by ContrastiveAttention)
    # ------------------------------------------------------------------

    def is_normal_report(self, report_text):
        """
        Returns True if BiomedCLIP considers this report to describe a
        normal/unremarkable chest X-ray. Used to build the CA normality pool.

        Uses zero-shot text-text similarity against normal vs abnormal anchors.
        """
        clean = self._clean_report(report_text)
        anchors = [_ANCHOR_PHRASES['normal'], _ANCHOR_PHRASES['abnormal']]
        texts = [clean] + anchors
        embs = biomedclip_encode_text(texts, device=self.biomedclip_device)
        report_emb = embs[0:1]          # [1, 512]
        anchor_embs = embs[1:]          # [2, 512]
        sims = torch.matmul(report_emb, anchor_embs.T).squeeze(0)  # [2]
        return sims[0].item() > sims[1].item()   # normal > abnormal


# =============================================================================
# 3. GCN Layer (unchanged)
# =============================================================================

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        stdv = 1. / math.sqrt(out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        out = (torch.matmul(adj, support) if x.dim() == 2
               else torch.matmul(adj.unsqueeze(0), support))
        return out + self.bias if self.bias is not None else out


# =============================================================================
# 4. KG Encoder — image-conditioned, anti-over-smoothing (unchanged)
# =============================================================================

class KnowledgeGraphEncoder(nn.Module):
    def __init__(self, num_nodes, node_types, d_model, d_visual,
                 num_gcn_layers=1, dropout=0.1, gcn_residual_alpha=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model   = d_model
        self.alpha     = gcn_residual_alpha

        self.node_embeddings = nn.Embedding(num_nodes, d_model)
        self.type_map  = {'anatomy': 0, 'abnormal': 1, 'normal': 2}
        self.register_buffer(
            'type_ids',
            torch.LongTensor([self.type_map[t] for t in node_types])
        )
        self.type_embeddings = nn.Embedding(3, d_model)
        self.gcn_layers = nn.ModuleList(
            [GraphConvolution(d_model, d_model) for _ in range(num_gcn_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)
        self.visual_to_node_gate = nn.Sequential(
            nn.Linear(d_visual, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, num_nodes),
        )
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        nn.init.xavier_uniform_(self.type_embeddings.weight)

    def forward(self, adj, fc_feats):
        B       = fc_feats.size(0)
        node_ids = torch.arange(self.num_nodes, device=adj.device)
        x = self.node_embeddings(node_ids) + self.type_embeddings(self.type_ids)
        for gcn in self.gcn_layers:
            residual = x
            out = F.relu(self.dropout(gcn(x, adj)))
            x   = self.alpha * out + (1 - self.alpha) * residual
        x = self.norm(x)
        gates    = torch.sigmoid(self.visual_to_node_gate(fc_feats))
        kg_feats = x.unsqueeze(0).expand(B, -1, -1) * gates.unsqueeze(-1)
        return kg_feats


# =============================================================================
# 5. Multi-Label Classifier (unchanged)
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
# 6. KG Cross-Attention (unchanged)
# =============================================================================

class KGCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k       = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.gate_linear   = nn.Linear(d_model * 2, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
        nn.init.constant_(self.gate_linear.bias, -2.0)

    def forward(self, decoder_hidden, kg_feats):
        B, L, D = decoder_hidden.shape
        residual = decoder_hidden
        x = self.norm(decoder_hidden)
        N = kg_feats.size(1)
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(kg_feats).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(kg_feats).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn   = self.dropout(F.softmax(scores, dim=-1))
        ctx    = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, D)
        ctx    = self.W_o(ctx)
        g      = torch.sigmoid(self.gate_linear(torch.cat([residual, ctx], dim=-1)))
        return residual + self.residual_scale * g * self.dropout(ctx)


# =============================================================================
# 7. KG Alignment Loss (unchanged)
# =============================================================================

class KGAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, kg_attention_weights, kg_labels):
        avg_attn = kg_attention_weights.mean(dim=1)
        return F.binary_cross_entropy(avg_attn.clamp(1e-7, 1 - 1e-7), kg_labels)
    