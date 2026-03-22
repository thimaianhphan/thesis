"""
Knowledge Graph Module for R2Gen

Implements:
- Medical Knowledge Graph construction from IU X-Ray annotations
  Based on: Zhang et al. "When Radiology Report Generation Meets Knowledge Graph" (AAAI 2020)
  
- Graph Convolutional Network (GCN) for KG encoding
  Based on: Kipf & Welling "Semi-Supervised Classification with GCNs" (ICLR 2017)
  
- KG-aware cross-attention for decoder fusion
  Based on: KiUT, Huang et al. (CVPR 2023) — Injected Knowledge Distiller concept
  
- Normal/Abnormal node separation
  Based on: DCG, Liang et al. "Divide and Conquer" (ACM MM 2024)
  
- Multi-label classification pretraining for KG nodes
  Based on: PPKED, Liu et al. (CVPR 2021) — two-stage training strategy
"""

import json
import re
import math
import copy
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Knowledge Graph Construction
#    Ref: Zhang et al. (AAAI 2020) — extract entities and co-occurrence edges
#    Ref: DCG (ACM MM 2024) — separate normal vs. abnormal nodes
# =============================================================================

# Predefined medical entities for chest X-ray domain (IU X-Ray)
# Organized into anatomy and finding categories following RadGraph schema
# (Jain et al., NeurIPS Datasets & Benchmarks 2021)

ANATOMY_ENTITIES = [
    'lung', 'lungs', 'heart', 'cardiac', 'mediastinum', 'mediastinal',
    'pleural', 'aorta', 'aortic', 'thoracic', 'diaphragm', 'rib', 'ribs',
    'spine', 'vertebral', 'costophrenic', 'hilum', 'hilar', 'trachea',
    'bronchial', 'pulmonary', 'chest', 'sternum', 'clavicle', 'shoulder',
    'abdomen', 'airspace', 'lobe', 'apex', 'base', 'vasculature',
    'vascular', 'silhouette', 'bony', 'osseous', 'skeletal',
]

FINDING_ENTITIES = {
    # Abnormal findings (disease-specific nodes per DCG)
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
    # Normal findings (disease-free nodes per DCG)
    'normal': [
        'normal', 'clear', 'unremarkable', 'stable', 'intact', 'midline',
        'symmetric', 'preserved', 'adequate', 'appropriate', 'satisfactory',
        'no acute', 'within normal', 'negative', 'free',
    ]
}

# Synonym mapping to consolidate entities → canonical clinical terms
# Ref: RadLex ontology for standardized terminology
SYNONYM_MAP = {
    'lungs': 'lung', 'cardiac': 'heart', 'mediastinal': 'mediastinum',
    'aortic': 'aorta', 'hilar': 'hilum', 'ribs': 'rib',
    'opacities': 'opacity', 'effusions': 'effusion',
    'infiltrates': 'infiltrate', 'nodules': 'nodule',
    'enlarged': 'enlargement', 'tortuous': 'tortuosity',
    'hyperinflated': 'hyperinflation',
}


class KnowledgeGraphBuilder:
    """
    Builds a medical knowledge graph from IU X-Ray report annotations.
    
    Graph structure (following Zhang et al., AAAI 2020):
    - Nodes: Medical entities (anatomy + findings)
    - Edges: Co-occurrence within same report (weighted by frequency)
    
    Enhanced with DCG (ACM MM 2024):
    - Separate node types: anatomy, abnormal_finding, normal_finding
    - This separation enables the decoder to focus on abnormal nodes
    """
    
    def __init__(self, ann_path, dataset_name='iu_xray', co_occur_threshold=3):
        """
        Args:
            ann_path: Path to annotation.json
            dataset_name: 'iu_xray' or 'mimic_cxr'
            co_occur_threshold: Minimum co-occurrence count to create an edge
        """
        self.ann_path = ann_path
        self.dataset_name = dataset_name
        self.co_occur_threshold = co_occur_threshold
        
        # Build entity vocabulary
        self.all_finding_words = (FINDING_ENTITIES['abnormal'] + 
                                  FINDING_ENTITIES['normal'])
        self.all_entity_words = ANATOMY_ENTITIES + self.all_finding_words
        
    def _clean_report(self, report):
        """Clean report text (matches R2Gen's tokenizer cleaning)."""
        report = report.replace('..', '.').replace('..', '.').strip().lower()
        report = re.sub('[.,?;*!%^&_+():\\-\\[\\]{}]', ' ', report)
        report = re.sub('\\s+', ' ', report)
        return report
    
    def _extract_entities(self, report_text):
        """Extract medical entities from a single report."""
        clean = self._clean_report(report_text)
        words = clean.split()
        
        found_entities = set()
        for word in words:
            # Canonicalize via synonym map
            canonical = SYNONYM_MAP.get(word, word)
            if canonical in self.all_entity_words:
                found_entities.add(canonical)
                
        # Also check bigrams for multi-word entities like "no acute"
        for i in range(len(words) - 1):
            bigram = words[i] + ' ' + words[i+1]
            if bigram in self.all_finding_words:
                found_entities.add(bigram)
                
        return found_entities
    
    def _get_node_type(self, entity):
        """Classify node type following DCG (ACM MM 2024)."""
        if entity in ANATOMY_ENTITIES:
            return 'anatomy'
        elif entity in FINDING_ENTITIES['abnormal']:
            return 'abnormal'
        elif entity in FINDING_ENTITIES['normal']:
            return 'normal'
        else:
            return 'anatomy'  # default
    
    def build(self, split='train'):
        """
        Build the knowledge graph from training reports.
        
        Returns:
            node_list: List of entity names (ordered)
            node_types: List of node types ('anatomy', 'abnormal', 'normal')
            adjacency: numpy array [N, N] with co-occurrence weights
            node2idx: Dict mapping entity name → index
        """
        ann = json.loads(open(self.ann_path, 'r').read())
        reports = ann[split]
        
        # Step 1: Extract entities from all reports
        entity_counter = Counter()
        co_occurrence = defaultdict(int)
        
        for example in reports:
            entities = self._extract_entities(example['report'])
            for e in entities:
                entity_counter[e] += 1
            # Co-occurrence: all pairs in the same report
            for e1, e2 in combinations(sorted(entities), 2):
                co_occurrence[(e1, e2)] += 1
                
        # Step 2: Filter entities by frequency (keep those appearing >= threshold)
        min_entity_freq = 2
        valid_entities = {e for e, c in entity_counter.items() if c >= min_entity_freq}
        
        # Step 3: Build node list (anatomy first, then abnormal, then normal)
        # Following DCG's node ordering for clean separation
        anatomy_nodes = sorted([e for e in valid_entities if self._get_node_type(e) == 'anatomy'])
        abnormal_nodes = sorted([e for e in valid_entities if self._get_node_type(e) == 'abnormal'])
        normal_nodes = sorted([e for e in valid_entities if self._get_node_type(e) == 'normal'])
        
        node_list = anatomy_nodes + abnormal_nodes + normal_nodes
        node_types = (['anatomy'] * len(anatomy_nodes) + 
                      ['abnormal'] * len(abnormal_nodes) +
                      ['normal'] * len(normal_nodes))
        node2idx = {name: idx for idx, name in enumerate(node_list)}
        
        N = len(node_list)
        adjacency = np.zeros((N, N), dtype=np.float32)
        
        # Step 4: Fill adjacency with co-occurrence weights
        for (e1, e2), count in co_occurrence.items():
            if e1 in node2idx and e2 in node2idx and count >= self.co_occur_threshold:
                i, j = node2idx[e1], node2idx[e2]
                adjacency[i, j] = count
                adjacency[j, i] = count  # symmetric
        
        # Step 5: Normalize adjacency (add self-loops, then row-normalize)
        # Following Kipf & Welling (ICLR 2017): A_hat = D^{-1/2} (A + I) D^{-1/2}
        adjacency += np.eye(N)  # self-loops
        degree = adjacency.sum(axis=1)
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        adjacency = D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        print(f"[KG] Built knowledge graph: {N} nodes "
              f"({len(anatomy_nodes)} anatomy, {len(abnormal_nodes)} abnormal, "
              f"{len(normal_nodes)} normal)")
        print(f"[KG] Non-zero edges (after threshold): "
              f"{int((adjacency > 0).sum() - N)}")  # exclude self-loops
        
        return node_list, node_types, adjacency, node2idx
    
    def extract_labels_for_report(self, report_text, node_list, node2idx):
        """
        Extract multi-label ground truth for a report.
        Used in Stage 1 (multi-label classification) of the two-stage training 
        from Zhang et al. (AAAI 2020).
        
        Returns:
            labels: numpy array [N] with 1 for present entities, 0 otherwise
        """
        entities = self._extract_entities(report_text)
        labels = np.zeros(len(node_list), dtype=np.float32)
        for e in entities:
            if e in node2idx:
                labels[node2idx[e]] = 1.0
        return labels


# =============================================================================
# 2. Graph Convolutional Network
#    Ref: Kipf & Welling (ICLR 2017) — standard GCN layers
#    Applied to medical KG: Zhang et al. (AAAI 2020)
# =============================================================================

class GraphConvolution(nn.Module):
    """
    Single GCN layer: H' = sigma(A_hat * H * W)
    Ref: Kipf & Welling, ICLR 2017, Eq. (2)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj):
        """
        Args:
            x: Node features [N, in_features] or [B, N, in_features]
            adj: Normalized adjacency [N, N]
        Returns:
            output: [N, out_features] or [B, N, out_features]
        """
        support = torch.matmul(x, self.weight)
        if x.dim() == 3:
            # Batched: adj is [N,N], support is [B,N,out]
            output = torch.matmul(adj.unsqueeze(0), support)
        else:
            output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class KnowledgeGraphEncoder(nn.Module):
    """
    Multi-layer GCN encoder for the medical knowledge graph.
    
    Architecture follows Zhang et al. (AAAI 2020):
    - 2-layer GCN with ReLU activation and dropout
    - Input: node embeddings (learnable or pretrained)
    - Output: knowledge-enriched node representations
    
    Enhanced with DCG (ACM MM 2024):
    - Separate embedding initialization for anatomy/abnormal/normal nodes
    """
    
    def __init__(self, num_nodes, node_types, d_model, num_gcn_layers=2, 
                 dropout=0.1):
        """
        Args:
            num_nodes: Number of nodes in the KG
            node_types: List of 'anatomy'/'abnormal'/'normal' per node
            d_model: Hidden dimension (matches R2Gen's d_model=512)
            num_gcn_layers: Number of GCN layers (default: 2)
            dropout: Dropout rate
        """
        super(KnowledgeGraphEncoder, self).__init__()
        
        self.num_nodes = num_nodes
        self.d_model = d_model
        
        # Learnable node embeddings
        # Ref: Zhang et al. (AAAI 2020) uses GloVe; we use learnable for simplicity
        # Can be replaced with BioWordVec/PubMedBERT embeddings for better init
        self.node_embeddings = nn.Embedding(num_nodes, d_model)
        
        # Type embeddings (DCG: distinguish normal vs abnormal)
        self.type_map = {'anatomy': 0, 'abnormal': 1, 'normal': 2}
        type_ids = [self.type_map[t] for t in node_types]
        self.register_buffer('type_ids', torch.LongTensor(type_ids))
        self.type_embeddings = nn.Embedding(3, d_model)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            self.gcn_layers.append(GraphConvolution(d_model, d_model))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        nn.init.xavier_uniform_(self.type_embeddings.weight)
        
    def forward(self, adj):
        """
        Encode the knowledge graph.
        
        Args:
            adj: Normalized adjacency matrix [N, N] (on device)
            
        Returns:
            kg_feats: Knowledge-enriched node representations [N, d_model]
        """
        # Get node features: entity embedding + type embedding
        node_ids = torch.arange(self.num_nodes, device=adj.device)
        x = self.node_embeddings(node_ids) + self.type_embeddings(self.type_ids)
        
        # Apply GCN layers with residual connections
        for i, gcn in enumerate(self.gcn_layers):
            residual = x
            x = gcn(x, adj)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual  # residual connection
            
        x = self.norm(x)
        return x


# =============================================================================
# 3. Multi-Label Disease Classifier (Stage 1 pretraining)
#    Ref: Zhang et al. (AAAI 2020) — two-stage training
#    Ref: PPKED (CVPR 2021) — Posterior Knowledge Explorer
# =============================================================================

class KGMultiLabelClassifier(nn.Module):
    """
    Multi-label classifier for KG node prediction.
    
    Used in Stage 1 of two-stage training (Zhang et al., AAAI 2020):
    1. Train this classifier to predict which KG nodes are present
    2. Freeze classifier, train report generator with KG features
    
    Input: Visual features from ResNet-101
    Output: Sigmoid probabilities for each KG node
    """
    
    def __init__(self, visual_feat_dim, num_nodes, d_model=512):
        super(KGMultiLabelClassifier, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(visual_feat_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_nodes),
        )
        
    def forward(self, visual_feats):
        """
        Args:
            visual_feats: Pooled visual features [B, visual_feat_dim]
        Returns:
            logits: [B, num_nodes]
        """
        return self.projection(visual_feats)
    
    def get_loss(self, logits, labels):
        """Binary cross-entropy loss for multi-label classification."""
        return F.binary_cross_entropy_with_logits(logits, labels)


# =============================================================================
# 4. KG-Aware Cross-Attention (Decoder Fusion)
#    Ref: KiUT (CVPR 2023) — Injected Knowledge Distiller uses cross-attention
#    Ref: PPKED (CVPR 2021) — Multi-domain Knowledge Distiller
# =============================================================================

class KGCrossAttention(nn.Module):
    """
    Cross-attention between decoder hidden states and KG node representations.
    
    Injected into R2Gen's decoder as an additional sublayer.
    Following KiUT (Huang et al., CVPR 2023), this module lets the decoder
    attend to relevant KG nodes at each generation step, promoting the use
    of proper clinical terminology.
    
    Q = decoder hidden state
    K = V = GCN-encoded KG node representations
    """
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(KGCrossAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Gating mechanism from PPKED (CVPR 2021):
        # Adaptive gate to control how much KG knowledge to inject
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.)
    
    def forward(self, decoder_hidden, kg_feats):
        """
        Args:
            decoder_hidden: [B, seq_len, d_model]
            kg_feats: [N, d_model] — will be expanded to batch
            
        Returns:
            output: [B, seq_len, d_model] — KG-enhanced decoder hidden states
        """
        B, L, D = decoder_hidden.shape
        residual = decoder_hidden
        
        # Normalize input (pre-norm)
        x = self.norm(decoder_hidden)
        
        # Expand KG features to batch: [N, D] → [B, N, D]
        if kg_feats.dim() == 2:
            kg_feats = kg_feats.unsqueeze(0).expand(B, -1, -1)
        
        N = kg_feats.size(1)
        
        # Multi-head attention
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(kg_feats).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(kg_feats).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        context = self.W_o(context)
        
        # Gated residual (PPKED-style adaptive distilling)
        gate_input = torch.cat([residual, context], dim=-1)
        g = self.gate(gate_input)
        output = residual + g * self.dropout(context)
        
        return output


# =============================================================================
# 5. KG-Enhanced Vocabulary Bias
#    Supplements KG cross-attention by directly biasing output logits
#    toward clinical terms that are contextually relevant.
#    Ref: Inspired by Yang et al. "Knowledge Matters" (MedIA 2022) —
#         general knowledge enhances word-level predictions
# =============================================================================

class KGVocabularyBias(nn.Module):
    """
    At each decoding step, compute similarity between decoder hidden states
    and KG node embeddings, then add bias to vocabulary logits for tokens
    that correspond to KG entities.
    
    This directly addresses the "simple words" problem by boosting the 
    probability of clinical terms when the KG context is relevant.
    """
    
    def __init__(self, d_model, num_nodes, vocab_size, node_list, tokenizer):
        super(KGVocabularyBias, self).__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        
        # Projection from decoder hidden to KG space
        self.proj = nn.Linear(d_model, d_model)
        
        # Learnable bias scale
        self.bias_scale = nn.Parameter(torch.tensor(0.1))
        
        # Build mapping from KG nodes to vocabulary indices
        # node_to_vocab[i] = list of vocab indices for KG node i
        node_to_vocab = {}
        for idx, node_name in enumerate(node_list):
            # A node name might be a single token or multi-word
            tokens = node_name.split()
            vocab_ids = []
            for token in tokens:
                vid = tokenizer.get_id_by_token(token)
                if vid != tokenizer.get_id_by_token('<unk>'):
                    vocab_ids.append(vid)
            if vocab_ids:
                node_to_vocab[idx] = vocab_ids
        
        # Build sparse mapping matrix: [num_nodes, vocab_size]
        mapping = torch.zeros(num_nodes, vocab_size + 1)
        for node_idx, vocab_ids in node_to_vocab.items():
            for vid in vocab_ids:
                mapping[node_idx, vid] = 1.0
        self.register_buffer('node_vocab_mapping', mapping)
        
    def forward(self, decoder_hidden, kg_feats, logits):
        """
        Args:
            decoder_hidden: [B, seq_len, d_model]
            kg_feats: [N, d_model]
            logits: [B, seq_len, vocab_size+1] (before softmax)
            
        Returns:
            biased_logits: [B, seq_len, vocab_size+1]
        """
        # Project decoder hidden states
        proj_hidden = self.proj(decoder_hidden)  # [B, L, D]
        
        # Compute similarity with KG nodes: [B, L, N]
        if kg_feats.dim() == 2:
            kg_feats_exp = kg_feats.unsqueeze(0)  # [1, N, D]
        else:
            kg_feats_exp = kg_feats
            
        sim = torch.matmul(proj_hidden, kg_feats_exp.transpose(-2, -1))
        sim = sim / math.sqrt(self.d_model)
        node_weights = torch.sigmoid(sim)  # [B, L, N]
        
        # Map node weights to vocabulary bias: [B, L, vocab_size+1]
        vocab_bias = torch.matmul(node_weights, self.node_vocab_mapping)
        
        # Scale and add to logits
        biased_logits = logits + self.bias_scale * vocab_bias
        
        return biased_logits


# =============================================================================
# 6. KG Alignment Loss
#    Ref: Zhang et al. (AAAI 2020) — encourages decoder attention over KG nodes
#         to align with ground-truth findings in each report
# =============================================================================

class KGAlignmentLoss(nn.Module):
    """
    Alignment loss between decoder's attention distribution over KG nodes
    and the ground-truth multi-label vector.
    
    L_KG = BCE(mean_attention_over_steps, ground_truth_labels)
    
    This ensures the model attends to the correct KG nodes during generation.
    """
    
    def __init__(self):
        super(KGAlignmentLoss, self).__init__()
        
    def forward(self, kg_attention_weights, kg_labels):
        """
        Args:
            kg_attention_weights: [B, seq_len, num_nodes] — attention from 
                                  KG cross-attention module
            kg_labels: [B, num_nodes] — ground truth (0/1)
            
        Returns:
            loss: scalar
        """
        # Average attention across sequence steps
        avg_attn = kg_attention_weights.mean(dim=1)  # [B, num_nodes]
        
        # BCE loss
        loss = F.binary_cross_entropy(
            avg_attn.clamp(1e-7, 1 - 1e-7), 
            kg_labels
        )
        return loss