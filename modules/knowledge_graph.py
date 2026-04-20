"""
Knowledge Graph Module for R2Gen — v5: Revision 2 unigram-first node discovery

Changes from v4 (Revision 1):
  - Fix 2a: Unigram-first extraction; bigrams only if freq >= 30 reports, cap 10
  - Fix 2b: Aggressively expanded stopword set (modifiers, filler, position words)
  - Fix 2c: Revised anchors — bare nouns for anatomy, state-verb for normal
  - Fix 2d: Per-category confidence thresholds (anatomy=0.50, abnormal=0.55, normal=0.55)
  - Fix 2e: Morphological synonym map replaces BiomedCLIP cosine clustering
  - Fix 2f: Whitelist backfill for terms BiomedCLIP reliably mis-types
  - Fix 2g: max_nodes=80, no forced category balance
  - Fix 3:  Simplified label extraction — exact stem (unigram) or consecutive (bigram)

Kept from v4:
  - Fix 1 (KGCrossAttention with gated residual) — unchanged
  - Fix 4 integration checks — unchanged
  - BiomedCLIP text encoder — unchanged
  - GraphConvolution, KnowledgeGraphEncoder, KGMultiLabelClassifier — unchanged

NOTE: Checkpoints from prior runs are incompatible — retrain from scratch.
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


# Fix 2c: Revised anchors — bare nouns for anatomy, state-verb anchors for normal
_ANCHOR_EXAMPLES = {
    'anatomy': [
        "lung", "lungs", "heart", "cardiac", "pleura", "pleural",
        "diaphragm", "ribs", "mediastinum", "trachea", "aorta",
        "thoracic spine", "hilum", "costophrenic angle",
    ],
    'abnormal': [
        "there is a pleural effusion",
        "opacity in the lower lobe",
        "enlarged cardiac silhouette",
        "patchy consolidation",
        "a nodule is present",
        "pneumothorax with mediastinal shift",
        "fracture of the rib",
        "cardiomegaly",
    ],
    'normal': [
        "lungs are clear",
        "no focal consolidation",
        "heart size is normal",
        "unremarkable",
        "within normal limits",
        "no acute cardiopulmonary abnormality",
        "stable appearance",
    ],
}

# Per-category confidence thresholds (Fix 2d)
_CONF_THRESHOLD = {
    'anatomy': 0.50,
    'abnormal': 0.55,
    'normal': 0.55,
}

# Fix 2b: Aggressively expanded stopword set
_STOPWORDS = {
    # articles, pronouns, conjunctions, prepositions
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
    'without', 'note', 'exam', 'radiograph', 'radiographs',
    'change', 'new', 'old', 'now', 'interval', 'previously', 'however',
    'well', 'including', 'xxxx',
    # radiology reporting filler
    'image', 'images', 'view', 'views', 'study', 'examination',
    'patient', 'finding', 'findings', 'impression', 'comparison',
    'technique', 'history', 'indication', 'report',
    # vague qualifiers (not clinical content)
    'noted', 'seen', 'visible', 'visualized', 'identified', 'present',
    'appearance', 'appears', 'demonstrate', 'demonstrates', 'shows',
    'showing', 'reveals', 'suggests', 'suggestion', 'suspicious',
    'possibly', 'probably', 'likely', 'grossly', 'overall',
    'consistent', 'similar', 'unchanged', 'evidence',
    # modifiers that must NOT become KG nodes (caused mild→normal bug)
    'mild', 'moderate', 'severe', 'small', 'large', 'significant',
    'minimal', 'marked', 'subtle', 'prominent', 'patchy', 'dense',
    'focal', 'diffuse',
    # position/laterality (too generic for KG typing)
    'left', 'right', 'bilateral', 'unilateral', 'upper', 'lower',
    'anterior', 'posterior', 'lateral', 'medial', 'inferior', 'superior',
    # temporal/measurement filler
    'prior', 'previous', 'current', 'today', 'since', 'acute', 'chronic',
    # structure words that embed badly as isolated tokens
    'soft', 'tissue', 'structures', 'bony',
}

# Fix 2f: Backfill whitelists — force correct type if present at freq >= 50
_ANATOMY_BACKFILL = {
    'lung', 'heart', 'pleural', 'diaphragm', 'rib', 'mediastinum',
    'cardiac', 'thoracic', 'pulmonary', 'aorta', 'hilum',
    'vasculature', 'silhouette',
}

_NORMAL_BACKFILL = {
    'normal', 'clear', 'unremarkable', 'stable', 'intact', 'preserved',
    'symmetric', 'midline', 'satisfactory', 'adequate', 'appropriate',
    'negative', 'free',
}

_ABNORMAL_BACKFILL = {
    'cardiomegaly', 'pneumothorax', 'atelectasis', 'edema',
    'effusion', 'consolidation', 'opacity', 'infiltrate',
    'pneumonia', 'nodule', 'mass', 'fracture',
}

_BACKFILL_FREQ_MIN = 50


# =============================================================================
# 2. KnowledgeGraphBuilder
# =============================================================================

class KnowledgeGraphBuilder:
    def __init__(self, ann_path, dataset_name='iu_xray',
                 co_occur_threshold=3,
                 min_term_freq=5,
                 max_nodes=80,
                 biomedclip_device='cpu'):
        self.ann_path           = ann_path
        self.dataset_name       = dataset_name
        self.co_occur_threshold = co_occur_threshold
        self.min_term_freq      = min_term_freq
        self.max_nodes          = max_nodes
        self.biomedclip_device  = biomedclip_device

        self._node_list  = None
        self._node2idx   = None
        self.synonym_map = {}
        self._prototypes = None

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

    def _build_morphological_synonyms(self, typed_terms):
        """
        Group morphological variants by stem. Keep highest-freq as canonical.
        Only merges within the same type — cross-type merges are rejected.
        Returns: (canonical_terms dict, synonym_map dict)
        """
        groups = {}
        for term, info in typed_terms.items():
            s = _stem(term)
            groups.setdefault(s, []).append((term, info['freq']))

        synonym_map    = {}
        canonical_terms = {}
        for stem, members in groups.items():
            members.sort(key=lambda x: -x[1])
            canonical = members[0][0]
            canonical_terms[canonical] = typed_terms[canonical]
            for term, _ in members[1:]:
                if typed_terms[term]['type'] == typed_terms[canonical]['type']:
                    synonym_map[term] = canonical
                else:
                    # Cross-type: keep as separate node
                    canonical_terms[term] = typed_terms[term]
        return canonical_terms, synonym_map

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

        # ---- Phase 1: corpus scan — unigram-first ----
        print(f"[KG] Phase 1: scanning {len(reports)} reports...")
        unigram_freq = Counter()
        bigram_freq  = Counter()

        for example in reports:
            clean = self._clean_report(example['report'])
            words = clean.split()
            seen_uni = set()
            seen_bi  = set()
            for w in words:
                if (len(w) >= 4
                        and w not in _STOPWORDS
                        and w.isalpha()):
                    seen_uni.add(w)
            for k in range(len(words) - 1):
                bi = words[k] + ' ' + words[k + 1]
                seen_bi.add(bi)
            for w in seen_uni:
                unigram_freq[w] += 1
            for b in seen_bi:
                bigram_freq[b] += 1

        # Unigrams: freq >= min_term_freq
        retained_uni = {
            t for t, f in unigram_freq.items()
            if f >= self.min_term_freq
        }

        # Bigrams: both tokens alphabetic, not stopwords, freq >= 30; cap at 10
        _BIGRAM_MIN_FREQ = 30
        _BIGRAM_CAP      = 10
        valid_bigrams = []
        for bi, f in bigram_freq.most_common():
            if f < _BIGRAM_MIN_FREQ:
                break
            w1, w2 = bi.split()
            if (w1 not in _STOPWORDS and w2 not in _STOPWORDS
                    and w1.isalpha() and w2.isalpha()
                    and len(w1) >= 4 and len(w2) >= 4):
                valid_bigrams.append(bi)
            if len(valid_bigrams) >= _BIGRAM_CAP:
                break

        all_candidates = sorted(
            list(retained_uni) + valid_bigrams,
            key=lambda t: -(unigram_freq.get(t, bigram_freq.get(t, 0)))
        )

        # Pre-BiomedCLIP cap
        cap = self.max_nodes * 5
        if len(all_candidates) > cap:
            all_candidates = all_candidates[:cap]

        def _get_freq(t):
            return unigram_freq.get(t, bigram_freq.get(t, 0))

        print(f"[KG] Phase 1 complete: {len(all_candidates)} candidates "
              f"({len(retained_uni)} unigrams, {len(valid_bigrams)} bigrams)")

        if not all_candidates:
            raise RuntimeError(
                "[KG] No valid terms found — lower min_term_freq or check data.")

        # ---- Phase 2: BiomedCLIP typing ----
        print(f"[KG] Phase 2: BiomedCLIP typing of {len(all_candidates)} terms...")
        P         = self._get_prototypes()                                 # [3, 512]
        term_embs = biomedclip_encode_text(
            all_candidates, device=self.biomedclip_device)                 # [N, 512]

        sims  = torch.matmul(term_embs, P.T)                              # [N, 3]
        probs = F.softmax(sims / 0.05, dim=-1)

        type_names = ['anatomy', 'abnormal', 'normal']
        type_thresholds = [
            _CONF_THRESHOLD['anatomy'],
            _CONF_THRESHOLD['abnormal'],
            _CONF_THRESHOLD['normal'],
        ]

        typed_terms = {}   # term -> {type, prob, freq, emb}
        dropped_low = 0
        dropped_amb = 0

        for i, term in enumerate(all_candidates):
            p_vec = probs[i]
            above = [(p_vec[j].item(), type_names[j])
                     for j in range(3)
                     if p_vec[j].item() >= type_thresholds[j]]

            if len(above) == 0:
                dropped_low += 1
                continue
            # Pick highest prob among those that passed their threshold
            best_prob, best_type = max(above, key=lambda x: x[0])
            typed_terms[term] = {
                'type': best_type,
                'prob': best_prob,
                'freq': _get_freq(term),
                'emb':  term_embs[i],
                'origin': 'typed',
            }

        print(f"[KG] Confidence thresholding: kept {len(typed_terms)}, "
              f"dropped {dropped_low} below threshold, "
              f"{dropped_amb} ambiguous")

        # ---- Phase 3: Backfill whitelists ----
        backfill_sets = [
            (_ANATOMY_BACKFILL,  'anatomy'),
            (_NORMAL_BACKFILL,   'normal'),
            (_ABNORMAL_BACKFILL, 'abnormal'),
        ]
        for bset, btype in backfill_sets:
            for term in bset:
                freq = _get_freq(term)
                if freq < _BACKFILL_FREQ_MIN:
                    continue
                if term in typed_terms and typed_terms[term]['type'] == btype:
                    continue  # already correctly typed

                # Determine BiomedCLIP conf if available (for logging)
                if term in all_candidates:
                    idx = all_candidates.index(term)
                    biomedclip_conf = probs[idx, type_names.index(btype)].item()
                    conf_str = f"BiomedCLIP conf={biomedclip_conf:.2f}"
                    if term in typed_terms:
                        old_type = typed_terms[term]['type']
                        conf_str += f", was {old_type}"
                else:
                    conf_str = "not in candidates"

                print(f"[KG] Backfilled {btype}: '{term}' "
                      f"(freq={freq}, {conf_str})")

                if term in all_candidates:
                    idx = all_candidates.index(term)
                    emb = term_embs[idx]
                else:
                    # Encode on the fly
                    emb = biomedclip_encode_text(
                        [term], device=self.biomedclip_device)[0]

                typed_terms[term] = {
                    'type':   btype,
                    'prob':   1.0,
                    'freq':   freq,
                    'emb':    emb,
                    'origin': 'backfill',
                }

        if not typed_terms:
            raise RuntimeError(
                "[KG] All terms dropped — lower thresholds or verify data.")

        # ---- Phase 4: Morphological synonym map ----
        canonical_terms, self.synonym_map = \
            self._build_morphological_synonyms(typed_terms)

        print(f"[KG] Morphological synonyms: {len(self.synonym_map)} entries, "
              f"{len(canonical_terms)} unique terms")

        # ---- Phase 5: Score and cap ----
        # Backfilled terms always kept; remaining sorted by freq × prob
        backfilled = {t: info for t, info in canonical_terms.items()
                      if info['origin'] == 'backfill'}
        scored = [(t, info) for t, info in canonical_terms.items()
                  if info['origin'] != 'backfill']
        scored.sort(key=lambda x: -(x[1]['freq'] * x[1]['prob']))

        remaining_cap = max(0, self.max_nodes - len(backfilled))
        scored = scored[:remaining_cap]

        surviving = list(backfilled.items()) + scored

        node_list  = [t for t, _ in surviving]
        node_info  = [info for _, info in surviving]
        node_types = [info['type'] for info in node_info]
        node2idx   = {name: idx for idx, name in enumerate(node_list)}
        N          = len(node_list)

        # ---- Co-occurrence pass ----
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
                i, j      = node2idx[t1], node2idx[t2]
                adj[i, j] = count
                adj[j, i] = count

        adj += np.eye(N)
        deg  = adj.sum(axis=1)
        d_inv = np.power(deg, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        adj = np.diag(d_inv) @ adj @ np.diag(d_inv)

        # Cache
        self._node_list = node_list
        self._node2idx  = node2idx

        # ---- Print final node list ----
        by_type = defaultdict(list)
        for t, info in surviving:
            by_type[info['type']].append((t, info))
        n_anat = len(by_type['anatomy'])
        n_abnl = len(by_type['abnormal'])
        n_norm = len(by_type['normal'])
        print(f"\n[KG] Final graph: {N} nodes  "
              f"({n_anat} anatomy, {n_abnl} abnormal, {n_norm} normal)  "
              f"synonym map: {len(self.synonym_map)} entries")
        for cat in ('anatomy', 'abnormal', 'normal'):
            for t, info in by_type[cat]:
                tag = f"[{info['origin']}]" if info['origin'] == 'backfill' else ''
                print(f"    [{cat:8s}]  {t:<35s}  "
                      f"freq={info['freq']:5d}  conf={info['prob']:.2f}  {tag}")
        if self.synonym_map:
            print("[KG] Synonyms (morphological only):")
            for src, dst in sorted(self.synonym_map.items()):
                print(f"    {src} → {dst}")
        print()

        return node_list, node_types, adj, node2idx

    # ------------------------------------------------------------------
    # Fix 3: Simplified label extraction — exact stem (unigram) or consecutive (bigram)
    # ------------------------------------------------------------------

    def extract_labels_for_report(self, report_text, node_list, node2idx):
        """
        Binary label vector [N].
        Applies synonym map first, then stemming for unigrams.
        Bigram nodes require consecutive stem match in the token stream.
        """
        clean  = self._clean_report(report_text)
        tokens = clean.split()
        # Apply synonym map to report tokens
        tokens = [self.synonym_map.get(t, t) for t in tokens]
        stems  = [_stem(t) for t in tokens]
        stem_set = set(stems)

        labels = np.zeros(len(node_list), dtype=np.float32)
        for i, node in enumerate(node_list):
            if ' ' in node:
                node_stems = [_stem(t) for t in node.split()]
                for j in range(len(stems) - len(node_stems) + 1):
                    if stems[j:j + len(node_stems)] == node_stems:
                        labels[i] = 1.0
                        break
            else:
                if _stem(node) in stem_set:
                    labels[i] = 1.0
        return labels

    def is_normal_report(self, report_text):
        """
        True if BiomedCLIP prototype similarity: normal > abnormal.
        """
        clean   = self._clean_report(report_text)
        P       = self._get_prototypes()
        rep_emb = biomedclip_encode_text([clean], device=self.biomedclip_device)
        sims    = torch.matmul(rep_emb, P.T).squeeze(0)
        return sims[2].item() > sims[1].item()


# =============================================================================
# 3. GCN Layer — global [N, N] adjacency
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
# 4. KnowledgeGraphEncoder — image-conditioned GCN, global adj
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
# 5. Multi-label classifier
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
# 6. KG Cross-Attention (Fix 1 — gated residual, unchanged from v4)
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
# 7. KG Alignment Loss
# =============================================================================

class KGAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, kg_attention_weights, kg_labels):
        avg_attn = kg_attention_weights.mean(dim=1)
        return F.binary_cross_entropy(avg_attn.clamp(1e-7, 1 - 1e-7), kg_labels)
