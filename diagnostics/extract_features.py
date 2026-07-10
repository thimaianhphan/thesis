"""
Phase A1 — Frozen feature extraction for the AE-vs-ResNet linear-probe diagnostic.

Extracts GAP-pooled visual features from a FROZEN encoder for every image in a
split, aligned to per-study KG findings labels, and saves them to .npz for
diagnostics/probe_and_rank.py.

  --encoder ae      ConvEncoder (modules/autoencoder.py) warm from ae_encoder.pth,
                    GAP over [B,256,14,14]  ->  D = 256
  --encoder resnet  torchvision ResNet-50 (ImageNet), children[:-2] + AvgPool2d,
                    ->  D = 2048            (the RRG reference backbone)

CRITICAL — preprocessing fidelity: we use the *exact* eval transform from
modules/dataloaders.py (Resize((224,224)) -> ToTensor -> Normalize(ImageNet)),
identically for BOTH encoders. The ConvAE was trained on ImageNet-normalized
inputs (see ConvDecoder docstring), and ResNet expects ImageNet stats, so the
shared transform is correct for both. We deliberately use the deterministic EVAL
transform (no RandomCrop/Flip) for ALL splits so features are reproducible. A
preprocessing mismatch would silently tank probe AUC and cause a false negative.

View handling (Phase 0.2): IU X-ray studies have 2 images; labels are per-study.
--view all (default) extracts BOTH images, each carrying its study label (fair to
both encoders, needs no view metadata). --view first restricts to image index 0
(≈ frontal under R2Gen's ordering convention, per-study unverified).

Labels: we save BOTH the RAW pipeline labels (y_raw, bit-identical to Stage-1
get_kg_labels) AND the negation-aware labels (y_negaware) so the probe can be run
either way without re-extraction. Phase 0.1 found ~71% of RAW abnormal positives
are negation artifacts.

HANDOFF: real extraction loads real images + real ckpt + wants a GPU -> run on the
remote machine. Locally, only `--smoke` (synthetic tensors, CPU) is run.

Examples (remote GPU):
    python diagnostics/extract_features.py --encoder ae     --split train --out feats/ae_train.npz
    python diagnostics/extract_features.py --encoder resnet  --split val   --out feats/rn_val.npz
Local smoke test (CPU, synthetic):
    python diagnostics/extract_features.py --smoke
"""

import argparse
import json
import os
import sys

import numpy as np

# Make repo root + this dir importable regardless of CWD.
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for p in (_ROOT, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

import kg_labels as K  # torch-free label utilities


# ----- ImageNet eval transform, mirrored from modules/dataloaders.py (val) ----
def build_eval_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def build_encoder(encoder, ckpt, resnet_arch='resnet50'):
    """Build a FROZEN encoder via the repo's VisualExtractor dispatcher so feature
    computation is bit-identical to the RRG pipeline. Returns (module, D)."""
    from types import SimpleNamespace
    from modules.visual_extractor import VisualExtractor
    if encoder == 'ae':
        args = SimpleNamespace(
            visual_extractor='autoencoder',
            autoencoder_ckpt=ckpt or 'artifacts/ae_encoder.pth',
            freeze_visual_extractor=True,
            d_vf=256,
        )
        D = 256
    elif encoder == 'resnet':
        args = SimpleNamespace(
            visual_extractor=resnet_arch,
            visual_extractor_pretrained=True,
            freeze_visual_extractor=True,
            d_vf=2048,
        )
        D = 2048
    else:
        raise ValueError(encoder)
    ve = VisualExtractor(args)
    ve.eval()
    for p in ve.parameters():
        p.requires_grad_(False)
    return ve, D


def _fc_feats(ve, batch):
    """Run dispatcher, return GAP fc_feats [B, D]."""
    _, fc = ve(batch)
    return fc


def extract(args):
    import torch
    from PIL import Image

    ann = json.loads(open(args.ann_path, 'r', encoding='utf-8').read())

    # Node vocab from the REAL builder when available (guarantees identical
    # ordering to training); fall back to the mirrored torch-free builder.
    try:
        from modules.knowledge_graph import KnowledgeGraphBuilder
        kgb = KnowledgeGraphBuilder(args.ann_path, 'iu_xray',
                                    co_occur_threshold=args.kg_co_occur_threshold)
        node_list, node_types, _, node2idx = kgb.build(split='train')
        raw_label = lambda rep: kgb.extract_labels_for_report(rep, node_list, node2idx)
    except Exception as e:
        print(f"[warn] using mirrored builder (real import failed: {e})")
        node_list, node_types, node2idx = K.build_nodes(ann, 'train')
        raw_label = lambda rep: K.labels_for_report(rep, node_list, node2idx)
    neg_label = lambda rep: K.labels_for_report_negaware(rep, node_list, node2idx)

    view_idx = {'all': [0, 1], 'first': [0], 'second': [1]}[args.view]
    transform = build_eval_transform()
    ve, D = build_encoder(args.encoder, args.ckpt, args.resnet_arch)
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    ve = ve.to(device)
    print(f"[extract] encoder={args.encoder} D={D} split={args.split} view={args.view} device={device}")

    studies = ann[args.split]
    feats, y_raw, y_neg, ids, paths, views = [], [], [], [], [], []
    buf_img, buf_meta = [], []

    def flush():
        if not buf_img:
            return
        batch = torch.stack(buf_img, 0).to(device)
        with torch.no_grad():
            fc = _fc_feats(ve, batch).cpu().numpy()
        assert fc.shape[1] == D, f"expected D={D}, got {fc.shape[1]}"
        feats.append(fc)
        for (iid, rep, rp, v) in buf_meta:
            y_raw.append(raw_label(rep))
            y_neg.append(neg_label(rep))
            ids.append(iid)
            paths.append(rp)
            views.append(v)
        buf_img.clear()
        buf_meta.clear()

    n_missing = 0
    for ex in studies:
        for v in view_idx:
            if v >= len(ex['image_path']):
                continue
            fp = os.path.join(args.image_dir, ex['image_path'][v])
            if not os.path.exists(fp):
                n_missing += 1
                continue
            img = Image.open(fp).convert('RGB')
            buf_img.append(transform(img))
            buf_meta.append((ex['id'], ex['report'], ex['image_path'][v], v))
            if len(buf_img) >= args.batch_size:
                flush()
    flush()

    X = np.concatenate(feats, 0).astype(np.float32)
    y_raw = np.stack(y_raw).astype(np.float32)
    y_neg = np.stack(y_neg).astype(np.float32)
    assert X.shape[0] == y_raw.shape[0] == len(ids), "row misalignment!"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X, y_raw=y_raw, y_negaware=y_neg,
        ids=np.array(ids), image_paths=np.array(paths), views=np.array(views),
        node_list=np.array(node_list), node_types=np.array(node_types),
        encoder=args.encoder, split=args.split, view=args.view, D=D,
    )
    print(f"[extract] saved {args.out}: X={X.shape} y={y_raw.shape} "
          f"(missing images skipped: {n_missing})")


# =============================================================================
# Smoke test — synthetic tensors, CPU, no real weights/images
# =============================================================================
def smoke():
    import torch
    from modules.autoencoder import ConvEncoder
    from modules.visual_extractor import ResNetVisualExtractor
    from types import SimpleNamespace
    print("[smoke] encoder output dims ...")

    # AE path: ConvEncoder directly (random init, no ckpt), mimic dispatcher GAP.
    enc = ConvEncoder().eval()
    x = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        fmap = enc(x)                       # [8,256,14,14]
        ae_fc = fmap.mean(dim=(2, 3))       # [8,256]
    assert fmap.shape == (8, 256, 14, 14), fmap.shape
    assert ae_fc.shape == (8, 256), ae_fc.shape
    print(f"  AE  fc_feats -> {tuple(ae_fc.shape)}  OK (==256)")

    # ResNet path: resnet50, pretrained=False (no download), via repo extractor.
    rn = ResNetVisualExtractor(SimpleNamespace(
        visual_extractor='resnet50', visual_extractor_pretrained=False)).eval()
    with torch.no_grad():
        _, rn_fc = rn(x)
    assert rn_fc.shape == (8, 2048), rn_fc.shape
    print(f"  RN  fc_feats -> {tuple(rn_fc.shape)}  OK (==2048)")

    # Label-alignment logic: toy ann, per-image expansion must carry study label.
    print("[smoke] label alignment ...")
    toy_ann = {'train': [
        {'id': 's1', 'report': 'The lungs are clear. No effusion.',
         'image_path': ['s1/0.png', 's1/1.png']},
        {'id': 's2', 'report': 'There is a large effusion and cardiomegaly.',
         'image_path': ['s2/0.png', 's2/1.png']},
    ]}
    node_list, node_types, node2idx = K.build_nodes(toy_ann, 'train', min_freq=1)
    # simulate --view all expansion
    ids, y_raw, y_neg = [], [], []
    for ex in toy_ann['train']:
        for v in [0, 1]:
            ids.append(ex['id'])
            y_raw.append(K.labels_for_report(ex['report'], node_list, node2idx))
            y_neg.append(K.labels_for_report_negaware(ex['report'], node_list, node2idx))
    y_raw = np.stack(y_raw); y_neg = np.stack(y_neg)
    assert ids == ['s1', 's1', 's2', 's2'], ids
    assert y_raw.shape[0] == 4 and y_raw.shape[1] == len(node_list)
    # both image rows of a study must be identical
    assert np.array_equal(y_raw[0], y_raw[1]) and np.array_equal(y_raw[2], y_raw[3])
    if 'effusion' in node2idx:
        ei = node2idx['effusion']
        assert y_raw[0, ei] == 1.0, "raw: negated effusion should be present"
        assert y_neg[0, ei] == 0.0, "negaware: negated effusion should be dropped"
        assert y_raw[2, ei] == 1.0 and y_neg[2, ei] == 1.0, "real effusion kept in both"
    print("  toy id->label mapping, per-image expansion, negation drop  OK")
    print("[smoke] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', choices=['ae', 'resnet'])
    ap.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    ap.add_argument('--view', default='all', choices=['all', 'first', 'second'])
    ap.add_argument('--out', default='feats/out.npz')
    ap.add_argument('--ckpt', default=None, help='AE ckpt (default artifacts/ae_encoder.pth)')
    ap.add_argument('--resnet_arch', default='resnet50')
    ap.add_argument('--image_dir', default='data/iu_xray/images/')
    ap.add_argument('--ann_path', default='data/iu_xray/annotation.json')
    ap.add_argument('--kg_co_occur_threshold', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--smoke', action='store_true', help='run CPU synthetic smoke test and exit')
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    if not args.encoder:
        ap.error('--encoder is required (ae|resnet) unless --smoke')
    extract(args)


if __name__ == '__main__':
    main()
