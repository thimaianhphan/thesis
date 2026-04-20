"""
Knowledge Graph Module for R2Gen — v4: Fixed BiomedCLIP node discovery

Fixes applied:
  - Fix 2a: Multi-example prototype anchors (replaces single-string _ANCHOR_PHRASES)
  - Fix 2b: Confidence threshold 0.6 — ambiguous terms dropped entirely
  - Fix 2c: Noun-biased bigram/trigram/unigram candidate extraction
  - Fix 2d: Synonym clustering via BiomedCLIP cosine > 0.92
  - Fix 2e: max_nodes=40, no forced 40/40/20 category balance
  - Fix 3:  Three-layer label extraction (synonym map + stemming + proximity)

NOTE: Checkpoints from prior runs (130+ node graphs) are incompatible —
      retrain from scratch after this change.
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
# Stemmer — NLTK PorterStemmer if available, else simple suffix stripper
# =============================================================================

try:
    from nltk.stem import PorterStemmer as _PorterStemmer
    _stemmer_inst = _PorterStemmer()
    def _stem(word):
        return _stemmer_inst.stem(word)
except ImportError:
    def _stem(word):
        if len(word) <= 3:
            return word
        for sfx in ('ations', 'nesses', 'ments', 'ation', 'ings', 'ness',
                    'ment', 'ies', 'ied', 'ier', 'iest', 'es', 'ed', 'ly', 's'):
            if word.endswith(sfx) and len(word) - len(sfx) >= 3:
                return word[:-len(sfx)]
        return word


# =============================================================================
# 1. BiomedCLIP Text Encoder — lazy-loaded, frozen
# =============================================================================

_BIOMEDCLIP_MODEL     = None
_BIOMEDCLIP_TOKENIZER = None
_BIOMEDCLIP_DEVICE    = None


def _load_biomedclip(device='cpu'):
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
        _BIOMEDCLIP_MODEL     = model
        _BIOMEDCLIP_TOKENIZER = tokenizer
        _BIOMEDCLIP_DEVICE    = device
        print("[KG] BiomedCLIP loaded.")
    return _BIOMEDCLIP_MODEL, _BIOMEDCLIP_TOKENIZER


@torch.no_grad()
def biomedclip_encode_text(phrases, device='cpu', batch_size=64):
    """Encode phrases with BiomedCLIP. Returns L2-normalised [N, 512]."""
    model, tokenizer = _load_biomedclip(device)
    all_embs = []
    for i in range(0, len(phrases), batch_size):
        batch   = phrases[i:i + batch_size]
        prompts = [f"a radiology image showing {p}" for p in batch]
        tokens  = tokenizer(prompts).to(device)
        embs    = model.encode_text(tokens)
        embs    = F.normalize(embs.float(), dim=-1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)   # [N, 512]


# Fix 2a: Multi-example prototype anchors
_ANCHOR_EXAMPLES = {
    'anatomy': [
        "the lungs and pleura are visualized",
        "the cardiac silhouette and mediastinum",
        "normal heart size and pulmonary vasculature",
        "the thoracic skeleton and soft tissues",
    ],
    'abnormal': [
        "there is a pleural effusion",
        "bilateral patchy opacity consistent with pneumonia",
        "enlarged cardiac silhouette with pulmonary edema",
        "focal consolidation in the lower lobe",
        "a large pneumothorax is present",
    ],
    'normal': [
        "lungs are well expanded and clear",
        "no focal consolidation effusion or pneumothorax",
        "no acute cardiopulmonary abnormality",
        "heart size and mediastinal contours are normal",
    ],
}


# =============================================================================
# 2. Candidate extraction helpers
# =============================================================================

# Fix 2c: Extended stopwords (added radiological noise words)
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
    'without', 'note', 'patient', 'exam', 'image', 'images', 'view',
    'views', 'radiograph', 'radiographs', 'chest', 'comparison',
    'change', 'new', 'old', 'prior', 'now', 'interval', 'since',
    'previously', 'however', 'well', 'including',
    # Added noise words (Fix 2c)
    'noted', 'seen', 'evidence', 'suggestion', 'suspicious', 'appearance',
    'findings', 'impression', 'bony', 'soft', 'tissue', 'structures',
    'consistent', 'similar', 'unchanged', 'stable', 'xxxx',
}

_MODIFIERS = {
    'small', 'large', 'mild', 'severe', 'bilateral', 'acute', 'chronic',
    'focal', 'diffuse', 'upper', 'lower', 'left', 'right', 'minimal',
    'moderate', 'marked', 'subtle', 'prominent', 'patchy', 'dense',
}


def _is_likely_noun(word):
    """Length > 3, not stopword, not adjective/adverb suffix."""
    return (len(word) > 3 and word not in _STOPWORDS
            and not word.endswith(('ly', 'ed', 'ing')))


def _is_modifier(word):
    """Known modifier or adjectival suffix."""
    return (word in _MODIFIERS
            or word.endswith(('al', 'ic', 'ive', 'ar'))
            or (word not in _STOPWORDS and len(word) <= 5))


# =============================================================================
# 3. KnowledgeGraphBuilder
# =============================================================================

class KnowledgeGraphBuilder:
    def __init__(self, ann_path, dataset_name='iu_xray',
                 co_occur_threshold=3,
                 min_term_freq=5,
                 max_nodes=40,          # Fix 2e: was 150
                 biomedclip_device='cpu'):
        self.ann_path           = ann_path
        self.dataset_name       = dataset_name
        self.co_occur_threshold = co_occur_threshold
        self.min_term_freq      = min_term_freq
        self.max_nodes          = max_nodes
        self.biomedclip_device  = biomedclip_device

        self._node_list  = None
        self._node2idx   = None
        self.synonym_map = {}        # Fix 2d: populated by build()
        self._prototypes = None      # cached prototype anchors [3, 512]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean_report(self, report):
        report = report.lower().strip()
        report = report.replace('..', '.').replace('..', '.')
        report = re.sub(r'[.,?;*!%^&_+():\-\[\]{}]', ' ', report)
        report = re.sub(r'\s+', ' ', report)
        return report

    def _get_prototypes(self):
        """Compute (or return cached) BiomedCLIP prototype vectors [3, 512]."""
        if self._prototypes is None:
            print("[KG] Computing prototype anchors from multi-example sets...")
            protos = []
            for cat in ('anatomy', 'abnormal', 'normal'):
                embs  = biomedclip_encode_text(
                    _ANCHOR_EXAMPLES[cat], device=self.biomedclip_device)
                proto = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
                protos.append(proto)
            self._prototypes = torch.cat(protos, dim=0)   # [3, 512]
            print("[KG] Prototype anchors ready.")
        return self._prototypes

    # Fix 2c: Noun-biased candidate extraction
    def _extract_candidates(self, report_text):
        """
        Returns (set of candidate terms, set of head nouns from bigrams).
        Bigrams : w1 w2  — w2 is likely noun, w1 not a stopword
        Trigrams: w1 w2 w3  — w1/w2 are modifiers, w3 is likely noun
        Unigrams: tentative; filtered further in build()
        """
        clean  = self._clean_report(report_text)
        words  = clean.split()
        candidates  = set()
        head_nouns  = set()

        for k in range(len(words) - 1):
            w1, w2 = words[k], words[k + 1]
            if w1 in _STOPWORDS:
                continue
            if not _is_likely_noun(w2):
                continue
            candidates.add(w1 + ' ' + w2)
            head_nouns.add(w2)

        for k in range(len(words) - 2):
            w1, w2, w3 = words[k], words[k + 1], words[k + 2]
            if not _is_likely_noun(w3):
                continue
            if not (_is_modifier(w1) and _is_modifier(w2)):
                continue
            candidates.add(w1 + ' ' + w2 + ' ' + w3)
            head_nouns.add(w3)

        for w in words:
            if len(w) > 3 and w not in _STOPWORDS:
                candidates.add(w)

        return candidates, head_nouns

    # ------------------------------------------------------------------
    # Public: build graph
    # ------------------------------------------------------------------

    def build(self, split='train'):
        """
        Build the knowledge graph from the training corpus.

        Returns:
            node_list  : list[str]
            node_types : list[str]  each in {anatomy, abnormal, normal}
            adjacency  : np.ndarray [N, N]  normalised
            node2idx   : dict {name: int}
        """
        ann     = json.loads(open(self.ann_path, 'r').read())
        reports = ann[split]

        # ---- Phase 1: corpus scan ----
        print(f"[KG] Phase 1: scanning {len(reports)} reports...")
        term_doc_freq       = Counter()
        bigram_to_head      = {}      # bigram string → head noun

        for example in reports:
            candidates, head_nouns = self._extract_candidates(example['report'])
            for t in candidates:
                term_doc_freq[t] += 1
            clean = self._clean_report(example['report'])
            words = clean.split()
            for k in range(len(words) - 1):
                w1, w2 = words[k], words[k + 1]
                if w1 not in _STOPWORDS and _is_likely_noun(w2):
                    bigram_to_head[w1 + ' ' + w2] = w2

        # Bigrams/trigrams: freq >= min_term_freq
        retained_multi = {
            t for t, f in term_doc_freq.items()
            if f >= self.min_term_freq and ' ' in t
        }
        # Head nouns from retained bigrams
        bigram_head_nouns = {
            bigram_to_head[b] for b in retained_multi
            if b.count(' ') == 1 and b in bigram_to_head
        }
        # Unigrams: must appear as head noun in a retained bigram AND freq >= 2x
        retained_uni = {
            t for t, f in term_doc_freq.items()
            if ' ' not in t
            and f >= 2 * self.min_term_freq
            and t in bigram_head_nouns
        }

        valid_terms = sorted(
            list(retained_multi | retained_uni),
            key=lambda t: -term_doc_freq[t]
        )

        # Cap before BiomedCLIP
        cap = self.max_nodes * 5
        if len(valid_terms) > cap:
            valid_terms = valid_terms[:cap]

        print(f"[KG] Phase 1 complete: {len(valid_terms)} candidates")

        if not valid_terms:
            raise RuntimeError(
                "[KG] No valid terms found — lower min_term_freq or check data.")

        # ---- Phase 2: BiomedCLIP typing ----
        print(f"[KG] Phase 2: BiomedCLIP typing of {len(valid_terms)} terms...")
        P         = self._get_prototypes()                               # [3, 512]
        term_embs = biomedclip_encode_text(
            valid_terms, device=self.biomedclip_device)                  # [N, 512]

        sims = torch.matmul(term_embs, P.T)                             # [N, 3]
        # Fix 2b: sharpen with low temperature, apply confidence threshold
        probs            = F.softmax(sims / 0.05, dim=-1)
        top_prob, top_idx = probs.max(dim=-1)

        type_names = ['anatomy', 'abnormal', 'normal']
        threshold  = 0.6

        surviving = []
        dropped   = 0
        for i, term in enumerate(valid_terms):
            p = top_prob[i].item()
            if p < threshold:
                dropped += 1
                continue
            surviving.append({
                'term': term,
                'type': type_names[top_idx[i].item()],
                'prob': p,
                'freq': term_doc_freq[term],
                'emb':  term_embs[i],
            })

        print(f"[KG] Confidence threshold ({threshold}): "
              f"kept {len(surviving)}, dropped {dropped} ambiguous terms")

        if not surviving:
            raise RuntimeError(
                "[KG] All terms dropped by confidence threshold — "
                "lower threshold or verify BiomedCLIP installation.")

        # ---- Fix 2d: Synonym clustering ----
        M = len(surviving)
        self.synonym_map = {}
        if M > 1:
            emb_stack  = torch.stack([s['emb'] for s in surviving])    # [M, 512]
            sim_matrix = torch.matmul(emb_stack, emb_stack.T)          # [M, M]
            is_canon   = [True] * M

            pairs = [(sim_matrix[i, j].item(), i, j)
                     for i in range(M) for j in range(i + 1, M)]
            pairs.sort(key=lambda x: -x[0])

            for sim_val, i, j in pairs:
                if sim_val <= 0.92:
                    break
                if not is_canon[i] or not is_canon[j]:
                    continue
                # Higher-frequency term is canonical
                if surviving[i]['freq'] >= surviving[j]['freq']:
                    canon, variant = i, j
                else:
                    canon, variant = j, i
                is_canon[variant] = False
                self.synonym_map[surviving[variant]['term']] = \
                    surviving[canon]['term']

            surviving = [s for k, s in enumerate(surviving) if is_canon[k]]

        print(f"[KG] Synonym merges: {len(self.synonym_map)}, "
              f"{len(surviving)} unique terms remaining")

        # ---- Fix 2e: Score and cap (no forced balance) ----
        surviving.sort(key=lambda s: -(s['freq'] * s['prob']))
        surviving = surviving[:self.max_nodes]

        node_list  = [s['term'] for s in surviving]
        node_types = [s['type'] for s in surviving]
        node2idx   = {name: idx for idx, name in enumerate(node_list)}
        N          = len(node_list)

        # ---- Co-occurrence pass (on final node set) ----
        print(f"[KG] Computing co-occurrence for {N} nodes...")
        co_occurrence = defaultdict(int)
        for example in reports:
            clean   = self._clean_report(example['report'])
            present = set()
            for node in node_list:
                if re.search(r'\b' + re.escape(node) + r'\b', clean):
                    present.add(node)
            for t1, t2 in combinations(sorted(present), 2):
                co_occurrence[(t1, t2)] += 1

        # ---- Adjacency matrix ----
        adj = np.zeros((N, N), dtype=np.float32)
        for (t1, t2), count in co_occurrence.items():
            if t1 in node2idx and t2 in node2idx and count >= self.co_occur_threshold:
                i, j       = node2idx[t1], node2idx[t2]
                adj[i, j]  = count
                adj[j, i]  = count

        # Symmetric normalisation: D^{-1/2} A D^{-1/2}
        adj    += np.eye(N)
        deg     = adj.sum(axis=1)
        d_inv   = np.power(deg, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        adj     = np.diag(d_inv) @ adj @ np.diag(d_inv)

        # Cache
        self._node_list = node_list
        self._node2idx  = node2idx

        # ---- Print final node list ----
        by_type = defaultdict(list)
        for s in surviving:
            by_type[s['type']].append(s)
        n_anat  = len(by_type['anatomy'])
        n_abnl  = len(by_type['abnormal'])
        n_norm  = len(by_type['normal'])
        print(f"\n[KG] Final graph: {N} nodes  "
              f"({n_anat} anatomy, {n_abnl} abnormal, {n_norm} normal)  "
              f"synonym map: {len(self.synonym_map)} entries")
        for cat in ('anatomy', 'abnormal', 'normal'):
            for s in by_type[cat]:
                print(f"    [{cat:8s}]  {s['term']:<35s}  "
                      f"freq={s['freq']:5d}  conf={s['prob']:.2f}")
        if self.synonym_map:
            print("[KG] Synonyms:")
            for src, dst in sorted(self.synonym_map.items()):
                print(f"    {src} → {dst}")
        print()

        return node_list, node_types, adj, node2idx

    # ------------------------------------------------------------------
    # Fix 3: Three-layer label extraction
    # ------------------------------------------------------------------

    def _apply_synonym_map(self, text):
        """Replace synonym variants with their canonical forms (longest first)."""
        for src, dst in sorted(self.synonym_map.items(),
                               key=lambda x: -len(x[0].split())):
            text = re.sub(r'\b' + re.escape(src) + r'\b', dst, text)
        return text

    def _stems_in_window(self, node_stems, report_stems, window=3):
        """
        True if node_stems appears as an in-order subsequence within any
        sliding window of size len(node_stems)+window over report_stems.
        """
        n        = len(node_stems)
        win_size = n + window
        for start in range(len(report_stems) - n + 1):
            segment = report_stems[start: start + win_size]
            idx     = 0
            for s in segment:
                if s == node_stems[idx]:
                    idx += 1
                    if idx == n:
                        return True
        return False

    def extract_labels_for_report(self, report_text, node_list, node2idx):
        """
        Fix 3: Binary label vector [N].
        Layer 1 — synonym canonicalization
        Layer 2 — stemming
        Layer 3 — multi-word proximity matching (window=3)
        """
        clean = self._clean_report(report_text)
        if self.synonym_map:
            clean = self._apply_synonym_map(clean)
        stems  = [_stem(t) for t in clean.split()]

        labels = np.zeros(len(node_list), dtype=np.float32)
        for i, node in enumerate(node_list):
            node_stems = [_stem(t) for t in node.split()]
            if len(node_stems) == 1:
                if node_stems[0] in stems:
                    labels[i] = 1.0
            else:
                if self._stems_in_window(node_stems, stems, window=3):
                    labels[i] = 1.0
        return labels

    def is_normal_report(self, report_text):
        """
        True if BiomedCLIP prototype similarity: normal > abnormal.
        Uses same prototype anchors as build() for consistency.
        """
        clean   = self._clean_report(report_text)
        P       = self._get_prototypes()                                # [3, 512]
        rep_emb = biomedclip_encode_text([clean],
                                         device=self.biomedclip_device) # [1, 512]
        sims    = torch.matmul(rep_emb, P.T).squeeze(0)                # [3]
        # indices: 0=anatomy, 1=abnormal, 2=normal
        return sims[2].item() > sims[1].item()


# =============================================================================
# 4. GCN Layer — global [N, N] adjacency
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
        """x: [N, d],  adj: [N, N]  → [N, d]"""
        support = torch.matmul(x, self.weight)
        out     = torch.matmul(adj, support)
        return out + self.bias if self.bias is not None else out


# =============================================================================
# 5. KnowledgeGraphEncoder — image-conditioned GCN, global adj
# =============================================================================

class KnowledgeGraphEncoder(nn.Module):
    def __init__(self, num_nodes, node_types, d_model, d_visual,
                 num_gcn_layers=1, dropout=0.1, gcn_residual_alpha=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model   = d_model
        self.alpha     = gcn_residual_alpha

        self.node_embeddings = nn.Embedding(num_nodes, d_model)
        type_map = {'anatomy': 0, 'abnormal': 1, 'normal': 2}
        self.register_buffer(
            'type_ids',
            torch.LongTensor([type_map.get(t, 0) for t in node_types])
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
        """
        adj      : [N, N]  global normalised adjacency (same across batch)
        fc_feats : [B, d_visual]
        Returns  : [B, N, d_model]
        """
        B        = fc_feats.size(0)
        node_ids = torch.arange(self.num_nodes, device=adj.device)
        x = (self.node_embeddings(node_ids)
             + self.type_embeddings(self.type_ids))                  # [N, d]
        for gcn in self.gcn_layers:
            residual = x
            out = F.relu(self.dropout(gcn(x, adj)))
            x   = self.alpha * out + (1 - self.alpha) * residual
        x = self.norm(x)                                             # [N, d]
        gates    = torch.sigmoid(self.visual_to_node_gate(fc_feats)) # [B, N]
        kg_feats = x.unsqueeze(0).expand(B, -1, -1) * gates.unsqueeze(-1)
        return kg_feats                                              # [B, N, d]


# =============================================================================
# 6. Multi-label classifier
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
# 7. KG Cross-Attention
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
        self.gate_linear    = nn.Linear(d_model * 2, 1)
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
# 8. KG Alignment Loss
# =============================================================================

class KGAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, kg_attention_weights, kg_labels):
        avg_attn = kg_attention_weights.mean(dim=1)
        return F.binary_cross_entropy(avg_attn.clamp(1e-7, 1 - 1e-7), kg_labels)
