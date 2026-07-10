"""
Post-extraction sanity check for Handoff 1 outputs (numpy-only, no GPU/torch).

Run this on the remote machine right after extract_features.py, BEFORE sending the
.npz files back. It catches the failure modes that make a probe silently useless:
degenerate/constant/NaN features (encoder not really loaded or preprocessing
mismatch), truncated splits (many missing images), wrong feature dim, and label
matrices that don't reflect the negation fix.

Usage:
    python diagnostics/check_features.py feats/ae_train.npz feats/ae_val.npz \
                                         feats/rn_train.npz feats/rn_val.npz
    python diagnostics/check_features.py --smoke      # self-test on a fake npz
"""

import argparse
import sys

import numpy as np

# expected study counts per split (from data/iu_xray/annotation.json)
SPLIT_STUDIES = {'train': 2069, 'val': 296, 'test': 590}
VIEW_MULT = {'all': 2, 'first': 1, 'second': 1}


def check_one(path):
    d = np.load(path, allow_pickle=True)
    enc = str(d['encoder']); split = str(d['split']); view = str(d['view'])
    X = d['X'].astype(np.float64)
    y_raw = d['y_raw']; y_neg = d['y_negaware']
    ids = d['ids']; views = d['views']; node_list = list(d['node_list'])
    node_types = list(d['node_types'])
    N, D = X.shape
    flags = []

    print(f"\n=== {path} ===")
    print(f"  encoder={enc}  split={split}  view={view}")
    print(f"  X={X.shape}  y_raw={y_raw.shape}  y_negaware={y_neg.shape}  "
          f"ids={len(ids)}  nodes={len(node_list)}")

    # ---- structural ----
    if not (N == y_raw.shape[0] == y_neg.shape[0] == len(ids) == len(views)):
        flags.append("ROW MISALIGNMENT across X / labels / ids / views")
    exp_D = {'ae': 256, 'resnet': 2048}.get(enc)
    if exp_D and D != exp_D:
        flags.append(f"D={D} but expected {exp_D} for encoder={enc}")
    if len(node_list) != y_raw.shape[1]:
        flags.append("node_list length != label columns")

    # expected N from split size x views (allow small shortfall for missing imgs)
    if split in SPLIT_STUDIES:
        exp_N = SPLIT_STUDIES[split] * VIEW_MULT.get(view, 1)
        miss = exp_N - N
        pct = 100 * miss / exp_N
        tag = "OK" if pct <= 2 else ("NOTE" if pct <= 10 else "RED FLAG")
        print(f"  N={N} vs expected {exp_N} (missing {miss}, {pct:.1f}%)  [{tag}]")
        if pct > 10:
            flags.append(f"{pct:.1f}% of images missing/skipped - truncated split?")

    # ---- feature health ----
    n_nan = int(np.isnan(X).sum()); n_inf = int(np.isinf(X).sum())
    if n_nan or n_inf:
        flags.append(f"non-finite features: {n_nan} NaN, {n_inf} Inf")
    var = X.var(0)
    dead = int((var < 1e-10).sum())
    zero_rows = int((np.abs(X).sum(1) < 1e-10).sum())
    dup_rows = N - len(np.unique(X.round(5), axis=0)) if N < 20000 else -1
    # cheap effective rank on standardized covariance
    Xc = X - X.mean(0, keepdims=True)
    lam = np.clip(np.linalg.eigvalsh((Xc.T @ Xc) / max(N - 1, 1)), 0, None)
    lam = lam[lam > 1e-12]
    p = lam / lam.sum()
    eff_rank = float(np.exp(-(p * np.log(p)).sum())) if len(lam) else 0.0
    print(f"  feat |mean|={np.abs(X).mean():.4f}  mean std={X.std(0).mean():.4f}  "
          f"range=[{X.min():.2f},{X.max():.2f}]")
    print(f"  dead features(var~0)={dead}/{D}  zero rows={zero_rows}  "
          f"dup rows={dup_rows}  eff_rank~={eff_rank:.1f}/{D}")
    if dead > 0.5 * D:
        flags.append(f"{dead}/{D} dead features - encoder likely not loaded / collapsed")
    if zero_rows > 0.02 * N:
        flags.append(f"{zero_rows} all-zero feature rows")
    if eff_rank < 2 and D > 4:
        flags.append(f"effective rank ~={eff_rank:.1f} - features nearly constant (broken extraction)")

    # ---- label health ----
    abn = [i for i, t in enumerate(node_types) if t == 'abnormal']
    raw_abn = float(y_raw[:, abn].sum()); neg_abn = float(y_neg[:, abn].sum())
    print(f"  positives/row raw={y_raw.sum(1).mean():.2f} negaware={y_neg.sum(1).mean():.2f}")
    print(f"  abnormal positives raw={int(raw_abn)} -> negaware={int(neg_abn)} "
          f"({100*(raw_abn-neg_abn)/max(raw_abn,1):.0f}% removed)")
    if (y_neg > y_raw).any():
        flags.append("negaware has positives absent from raw (labeler inconsistency)")
    if raw_abn > 0 and neg_abn / raw_abn > 0.6:
        flags.append("negation fix removed <40% of abnormal positives - expected ~70%; check labeler")

    if flags:
        print("  >>> ISSUES:")
        for f in flags:
            print(f"      - {f}")
    else:
        print("  >>> OK")
    return dict(path=path, enc=enc, split=split, view=view, D=D, N=N,
                node_list=node_list, flags=flags)


def cross_checks(reports):
    print("\n=== cross-file checks ===")
    issues = []
    # same node vocabulary everywhere
    nls = {tuple(r['node_list']) for r in reports}
    if len(nls) > 1:
        issues.append("node_list differs between files (vocab mismatch -> probe columns misaligned)")
    else:
        print("  node vocabulary identical across files  OK")
    # same view unit everywhere (fair AE-vs-ResNet comparison)
    if len({r['view'] for r in reports}) > 1:
        issues.append("mixed --view across files (unfair comparison)")
    else:
        print("  view unit identical across files  OK")
    # per encoder, train vs val same D
    for enc in {r['enc'] for r in reports}:
        Ds = {r['D'] for r in reports if r['enc'] == enc}
        if len(Ds) > 1:
            issues.append(f"encoder {enc} has inconsistent D across splits: {Ds}")
    if issues:
        print("  >>> ISSUES:")
        for i in issues:
            print(f"      - {i}")
    else:
        print("  all cross-file checks OK")
    return issues


def smoke():
    import os, tempfile
    print("[smoke] check_features on a fabricated npz ...")
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, 'ae_val.npz')
    rng = np.random.default_rng(0)
    N, D, C = 592, 256, 74
    X = rng.standard_normal((N, D)).astype(np.float32)
    y_raw = (rng.random((N, C)) > 0.9).astype(np.float32)
    y_neg = (y_raw * (rng.random((N, C)) > 0.5)).astype(np.float32)  # subset of raw
    node_types = ['anatomy'] * 29 + ['abnormal'] * 34 + ['normal'] * 11
    np.savez_compressed(
        p, X=X, y_raw=y_raw, y_negaware=y_neg,
        ids=np.array([f's{i//2}' for i in range(N)]),
        image_paths=np.array([f's{i//2}/{i%2}.png' for i in range(N)]),
        views=np.array([i % 2 for i in range(N)]),
        node_list=np.array([f'n{i}' for i in range(C)]),
        node_types=np.array(node_types),
        encoder='ae', split='val', view='all', D=D)
    r = check_one(p)
    assert not r['flags'], r['flags']
    print("[smoke] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', nargs='*')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        smoke(); return
    if not args.npz:
        ap.error('pass one or more .npz paths, or --smoke')
    reports = [check_one(p) for p in args.npz]
    any_flags = any(r['flags'] for r in reports)
    if len(reports) > 1:
        any_flags = cross_checks(reports) or any_flags
    print("\n" + ("*** RED FLAGS ABOVE - inspect before probing ***"
                  if any_flags else "*** ALL CHECKS PASSED - safe to probe / send back ***"))
    sys.exit(1 if any_flags else 0)


if __name__ == '__main__':
    main()
