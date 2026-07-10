"""
Phase 0.1 — Label spot-check for the KG findings-label pipeline.

Validates the multi-hot findings labels that (a) Part A's linear probe uses as its
target and (b) Part B's classification head trains on. Broken labels invalidate the
whole diagnostic and would corrupt Part B.

Torch-free (runs in the local no-GPU/no-torch env). Label logic is imported from
diagnostics/kg_labels.py, whose functions mirror modules/knowledge_graph.py
(KnowledgeGraphBuilder) verbatim, so labels here are bit-identical to Stage-1
training's get_kg_labels -> extract_labels_for_report.

Headline finding: the keyword matcher has NO negation handling, so a normal report
("no pneumothorax or pleural effusion is seen") sets the effusion / pneumothorax
ABNORMAL nodes to 1. This script quantifies that false-positive rate and prints
report/label pairs.

Run:
    python diagnostics/label_spotcheck.py --ann_path data/iu_xray/annotation.json
"""

import argparse
import json
import re
from collections import defaultdict

import numpy as np

import kg_labels as K  # local module (run from diagnostics/ or add to path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann_path', default='data/iu_xray/annotation.json')
    ap.add_argument('--split', default='train')
    ap.add_argument('--n_examples', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ann = json.loads(open(args.ann_path, 'r', encoding='utf-8').read())
    node_list, node_types, node2idx = K.build_nodes(ann, split='train')
    n_anat = node_types.count('anatomy')
    n_abnl = node_types.count('abnormal')
    n_norm = node_types.count('normal')
    print("=" * 78)
    print(f"KG node vocabulary (built on train, min_freq=2): N = {len(node_list)}")
    print(f"  anatomy={n_anat}  abnormal={n_abnl}  normal={n_norm}")
    print("=" * 78)

    reports = ann[args.split]
    L = K.label_matrix(reports, node_list, node2idx, node_types, negation_aware=False)
    Lneg = K.label_matrix(reports, node_list, node2idx, node_types, negation_aware=True)
    pos_counts = L.sum(0).astype(int)
    print(f"\n[{args.split}] {len(reports)} studies, label matrix {L.shape}")
    print(f"mean positives/study (raw): {L.sum(1).mean():.2f}   (negation-aware): {Lneg.sum(1).mean():.2f}")

    # ---- Negation audit on abnormal nodes ----
    abn_idx = [i for i, t in enumerate(node_types) if t == 'abnormal']
    total_abn_pos = int(L[:, abn_idx].sum())
    total_abn_neg = int((L[:, abn_idx].sum(0) - Lneg[:, abn_idx].sum(0)).sum())
    studies_with_fp = int((((L - Lneg)[:, abn_idx]) > 0).any(1).sum())
    print("\n" + "=" * 78)
    print("NEGATION AUDIT (abnormal labels set on NEGATED findings = false positives)")
    print("=" * 78)
    print(f"studies with >=1 negated-abnormal false positive : "
          f"{studies_with_fp}/{len(reports)}  ({100*studies_with_fp/len(reports):.1f}%)")
    print(f"abnormal-node positive labels total (raw)        : {total_abn_pos}")
    print(f"  of which set on a NEGATED finding (false pos)  : {total_abn_neg}  "
          f"({100*total_abn_neg/max(total_abn_pos,1):.1f}% of abnormal positives)")

    print("\nper-abnormal-node  raw_pos -> negaware_pos  (false-positive %):")
    for i in sorted(abn_idx, key=lambda k: -(L[:, k].sum() - Lneg[:, k].sum())):
        raw = int(L[:, i].sum())
        keep = int(Lneg[:, i].sum())
        fp = raw - keep
        print(f"  {node_list[i]:20s} {raw:5d} -> {keep:5d}   ({100*fp/max(raw,1):5.1f}% fp)")

    # ---- Concrete examples ----
    examples = []
    for e in reports:
        raw = K.labels_for_report(e['report'], node_list, node2idx)
        neg = K.labels_for_report_negaware(e['report'], node_list, node2idx)
        dropped = [node_list[i] for i in abn_idx if raw[i] > 0 and neg[i] == 0]
        kept = [node_list[i] for i in abn_idx if neg[i] > 0]
        if dropped:
            examples.append((e['id'], e['report'], kept, dropped))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(examples)
    print("\n" + "=" * 78)
    print(f"{min(args.n_examples, len(examples))} EXAMPLE STUDIES with negation-induced false positives")
    print("(report | KEPT abnormal (real) | DROPPED abnormal (negated -> spurious))")
    print("=" * 78)
    for iid, rep, kept, dropped in examples[:args.n_examples]:
        rep1 = re.sub(r'\s+', ' ', rep).strip()
        print(f"\n[{iid}]")
        print(f"  report : {rep1[:300]}")
        print(f"  KEEP   : {kept}")
        print(f"  DROP   : {dropped}   <-- spurious positives removed by negation fix")


if __name__ == '__main__':
    main()
