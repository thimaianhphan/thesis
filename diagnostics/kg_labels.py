"""
Torch-free KG findings-label utilities shared by the diagnostics + Part B.

Two labelers over the SAME node vocabulary:

  labels_for_report            -- RAW, bit-identical to modules/knowledge_graph.py
                                  KnowledgeGraphBuilder.extract_labels_for_report
                                  (keyword match, NO negation handling).
  labels_for_report_negaware   -- negation-aware: an abnormal node is set to 1
                                  only if it appears in a NON-negated context.
                                  Anatomy / normal nodes are unchanged.

Phase 0.1 found that ~71% of abnormal-node positives in the RAW labels are
negation artifacts ("no pleural effusion" -> effusion=1). The negation-aware
labeler is the proposed fix; keeping both lets Part A quantify how much of the
low abnormality signal is a LABEL problem vs a REPRESENTATION problem.

The node vocabulary (build_nodes) is mirrored from KnowledgeGraphBuilder.build
so node_list / ordering match training exactly. On the remote (torch) machine you
may instead import the real KnowledgeGraphBuilder for the vocab; this module is
provided so the local (no-torch) environment can compute labels too.
"""

import re
from collections import Counter
from itertools import combinations  # noqa: F401  (kept for parity with builder)

import numpy as np

# ---- Mirrored verbatim from modules/knowledge_graph.py (keep in sync) --------
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
    ],
}
SYNONYM_MAP = {
    'lungs': 'lung', 'cardiac': 'heart', 'mediastinal': 'mediastinum',
    'aortic': 'aorta', 'hilar': 'hilum', 'ribs': 'rib',
    'opacities': 'opacity', 'effusions': 'effusion',
    'infiltrates': 'infiltrate', 'nodules': 'nodule',
    'enlarged': 'enlargement', 'tortuous': 'tortuosity',
    'hyperinflated': 'hyperinflation',
}
ALL_FINDING_WORDS = FINDING_ENTITIES['abnormal'] + FINDING_ENTITIES['normal']
ALL_ENTITY_WORDS = ANATOMY_ENTITIES + ALL_FINDING_WORDS
ABNORMAL_CANON = frozenset(SYNONYM_MAP.get(w, w) for w in FINDING_ENTITIES['abnormal'])


def _clean_report(report):
    report = report.replace('..', '.').replace('..', '.').strip().lower()
    report = re.sub('[.,?;*!%^&_+():\\-\\[\\]{}]', ' ', report)
    report = re.sub('\\s+', ' ', report)
    return report


def _extract_entities(report_text):
    clean = _clean_report(report_text)
    words = clean.split()
    found = set()
    for word in words:
        canonical = SYNONYM_MAP.get(word, word)
        if canonical in ALL_ENTITY_WORDS:
            found.add(canonical)
    for i in range(len(words) - 1):
        bigram = words[i] + ' ' + words[i + 1]
        if bigram in ALL_FINDING_WORDS:
            found.add(bigram)
    return found


def get_node_type(entity):
    if entity in ANATOMY_ENTITIES:
        return 'anatomy'
    elif entity in FINDING_ENTITIES['abnormal']:
        return 'abnormal'
    elif entity in FINDING_ENTITIES['normal']:
        return 'normal'
    return 'anatomy'


def build_nodes(ann, split='train', min_freq=2):
    """Mirror of KnowledgeGraphBuilder.build node ordering (anat|abnl|norm, sorted)."""
    reports = ann[split]
    entity_counter = Counter()
    for example in reports:
        for e in _extract_entities(example['report']):
            entity_counter[e] += 1
    valid = {e for e, c in entity_counter.items() if c >= min_freq}
    anat = sorted([e for e in valid if get_node_type(e) == 'anatomy'])
    abnl = sorted([e for e in valid if get_node_type(e) == 'abnormal'])
    norm = sorted([e for e in valid if get_node_type(e) == 'normal'])
    node_list = anat + abnl + norm
    node_types = ['anatomy'] * len(anat) + ['abnormal'] * len(abnl) + ['normal'] * len(norm)
    node2idx = {name: idx for idx, name in enumerate(node_list)}
    return node_list, node_types, node2idx


def labels_for_report(report_text, node_list, node2idx):
    """RAW labels — bit-identical to KnowledgeGraphBuilder.extract_labels_for_report."""
    entities = _extract_entities(report_text)
    labels = np.zeros(len(node_list), dtype=np.float32)
    for e in entities:
        if e in node2idx:
            labels[node2idx[e]] = 1.0
    return labels


# ---- Negation-aware variant (the proposed fix) -------------------------------
_NEG_CUES = {'no', 'not', 'without', 'negative', 'free', 'resolved', 'absent',
             'cleared'}
_SCOPE_TERMINATORS = {'but', 'however', 'although', 'though', 'otherwise',
                      'except', 'with', 'shows', 'showing', 'demonstrates',
                      'demonstrating', 'reveals', 'revealing', 'redemonstrated'}


def _negated_abnormal(report_text):
    """Sentence-scoped set of canonical abnormal entities that are NEGATED."""
    lower = report_text.lower()
    negated = set()
    for sent in re.split(r'[.;:]', lower):
        toks = re.sub(r'[^a-z0-9 ]', ' ', sent).split()
        canon = [SYNONYM_MAP.get(t, t) for t in toks]
        postfix = bool(re.search(r'\bnot\b.*\b(seen|identified|present|appreciated|evident|noted)\b', sent))
        flag = False
        for i, t in enumerate(toks):
            if t in _SCOPE_TERMINATORS:
                flag = False
            if t in _NEG_CUES:
                flag = True
            if canon[i] in ABNORMAL_CANON and (flag or postfix):
                negated.add(canon[i])
    return negated


def labels_for_report_negaware(report_text, node_list, node2idx, node_types=None):
    """Negation-aware labels: abnormal nodes appearing only in a negated context
    are dropped. Anatomy / normal nodes are identical to the raw labeler."""
    entities = _extract_entities(report_text)
    negated = _negated_abnormal(report_text)
    labels = np.zeros(len(node_list), dtype=np.float32)
    for e in entities:
        if e not in node2idx:
            continue
        if e in ABNORMAL_CANON and e in negated:
            continue  # spurious: finding word only appears negated
        labels[node2idx[e]] = 1.0
    return labels


def label_matrix(reports, node_list, node2idx, node_types=None, negation_aware=False):
    """Stack labels for a list of {'report': ...} dicts -> [N, n_nodes] float32."""
    fn = labels_for_report_negaware if negation_aware else labels_for_report
    if negation_aware:
        return np.stack([fn(e['report'], node_list, node2idx, node_types) for e in reports])
    return np.stack([fn(e['report'], node_list, node2idx) for e in reports])
