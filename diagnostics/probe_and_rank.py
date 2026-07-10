"""
Phase A2 — Linear probe + effective-rank analysis on frozen features.

Pure numpy (no torch, no sklearn, no GPU/dataset) — runs on the .npz files from
extract_features.py. This is the number that gates Part B.

Linear probe:
  - Standardize features on TRAIN stats (mean-center + unit variance), apply to val.
    Mean-centering is the whole point: it removes the shared "this is a chest X-ray"
    DC component so the probe sees the residual that must carry findings.
  - Fit a single linear multi-label classifier (vectorized L2 logistic regression,
    one weight vector per finding). NO encoder fine-tuning.
  - Report on val: macro-AUC, micro-AUC, per-finding AP (sorted), per-finding
    positive counts, AND a breakdown by node type (anatomy/abnormal/normal).
    The ABNORMAL macro-AUC is the decision-gate number.

Effective rank / participation ratio (on mean-centered TRAIN covariance):
  - PR = (Σλ)² / Σλ²   ;  effective rank = exp(H), H = -Σ pᵢ log pᵢ, pᵢ = λᵢ/Σλ
  - top-10 eigenvalue share (variance concentration).

Emits one AE-vs-ResNet comparison table.

Usage (after npz returned from the remote GPU):
    python diagnostics/probe_and_rank.py \
        --ae_train feats/ae_train.npz --ae_val feats/ae_val.npz \
        --rn_train feats/rn_train.npz --rn_val feats/rn_val.npz \
        --label negaware
Local smoke test (synthetic arrays):
    python diagnostics/probe_and_rank.py --smoke
"""

import argparse
import numpy as np


# ----------------------------- metrics ---------------------------------------
def _rankdata(a):
    """Average ranks with tie handling (like scipy.stats.rankdata)."""
    a = np.asarray(a)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def auc_score(y, s):
    y = np.asarray(y).astype(int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = _rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def ap_score(y, s):
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind='mergesort')
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


# ----------------------------- probe -----------------------------------------
def standardize(Xtr, Xva):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xva - mu) / sd


def fit_logreg(Xtr, Ytr, l2=1e-2, lr=0.5, iters=500):
    """Vectorized multi-label L2 logistic regression (all findings at once)."""
    N, D = Xtr.shape
    C = Ytr.shape[1]
    W = np.zeros((D, C))
    b = np.zeros(C)
    for _ in range(iters):
        Z = Xtr @ W + b
        P = 1.0 / (1.0 + np.exp(-np.clip(Z, -30, 30)))
        G = (P - Ytr) / N
        W -= lr * (Xtr.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


def probe(train_npz, val_npz, label='negaware', verbose=True):
    tr = np.load(train_npz, allow_pickle=True)
    va = np.load(val_npz, allow_pickle=True)
    ykey = 'y_negaware' if label == 'negaware' else 'y_raw'
    Xtr, Ytr = tr['X'].astype(np.float64), tr[ykey].astype(np.float64)
    Xva, Yva = va['X'].astype(np.float64), va[ykey].astype(np.float64)
    node_list = list(tr['node_list'])
    node_types = list(tr['node_types'])
    D = Xtr.shape[1]

    Xtr_s, Xva_s = standardize(Xtr, Xva)
    W, b = fit_logreg(Xtr_s, Ytr)
    S = Xva_s @ W + b  # val scores [Nval, C]

    per = []
    for c in range(Ytr.shape[1]):
        ntr = int(Ytr[:, c].sum())
        nva = int(Yva[:, c].sum())
        per.append({
            'name': node_list[c], 'type': node_types[c],
            'n_train_pos': ntr, 'n_val_pos': nva,
            'auc': auc_score(Yva[:, c], S[:, c]),
            'ap': ap_score(Yva[:, c], S[:, c]),
        })

    def macro(types=None):
        v = [p['auc'] for p in per
             if not np.isnan(p['auc']) and (types is None or p['type'] in types)]
        return float(np.mean(v)) if v else np.nan

    # micro-AUC: pool all (finding, sample) pairs that are evaluable
    mask = np.array([not np.isnan(p['auc']) for p in per])
    if mask.any():
        yv = Yva[:, mask].ravel()
        sv = S[:, mask].ravel()
        micro = auc_score(yv, sv)
    else:
        micro = np.nan

    # ----- effective rank on mean-centered TRAIN covariance -----
    Xc = Xtr - Xtr.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    lam = np.clip(np.linalg.eigvalsh(cov), 0, None)
    lam = lam[lam > 1e-12]
    p = lam / lam.sum()
    PR = float((lam.sum() ** 2) / (lam ** 2).sum())
    H = float(-(p * np.log(p)).sum())
    eff_rank = float(np.exp(H))
    top10 = float(np.sort(lam)[::-1][:10].sum() / lam.sum())

    res = {
        'encoder': str(tr['encoder']), 'D': D, 'label': label,
        'macro_auc': macro(), 'micro_auc': micro,
        'macro_auc_anatomy': macro({'anatomy'}),
        'macro_auc_abnormal': macro({'abnormal'}),
        'macro_auc_normal': macro({'normal'}),
        'PR': PR, 'eff_rank': eff_rank, 'top10_share': top10,
        'per': per,
    }
    if verbose:
        _print_one(res)
    return res


def _print_one(res):
    print(f"\n=== {res['encoder']}  (D={res['D']}, labels={res['label']}) ===")
    print(f"  macro-AUC (all)      : {res['macro_auc']:.4f}")
    print(f"  micro-AUC            : {res['micro_auc']:.4f}")
    print(f"  macro-AUC anatomy    : {res['macro_auc_anatomy']:.4f}")
    print(f"  macro-AUC ABNORMAL   : {res['macro_auc_abnormal']:.4f}   <-- gate number")
    print(f"  macro-AUC normal     : {res['macro_auc_normal']:.4f}")
    print(f"  participation ratio  : {res['PR']:.2f} / {res['D']}")
    print(f"  effective rank       : {res['eff_rank']:.2f} / {res['D']}")
    print(f"  top-10 eig share     : {res['top10_share']:.3f}")
    per = sorted([p for p in res['per'] if not np.isnan(p['ap'])], key=lambda x: -x['ap'])
    print("  top per-finding AP (name/type/AP/AUC/val_pos):")
    for p in per[:12]:
        print(f"    {p['name']:20s} {p['type']:8s} AP={p['ap']:.3f} "
              f"AUC={p['auc']:.3f} vpos={p['n_val_pos']}")


def compare(results, label):
    print("\n" + "=" * 74)
    print(f"AE vs ResNet — comparison table (labels = {label})")
    print("=" * 74)
    hdr = f"{'metric':24s}" + "".join(f"{r['encoder']:>16s}" for r in results)
    print(hdr)
    print("-" * len(hdr))
    rows = [
        ('D (feature dim)', lambda r: f"{r['D']}"),
        ('macro-AUC (all)', lambda r: f"{r['macro_auc']:.4f}"),
        ('micro-AUC', lambda r: f"{r['micro_auc']:.4f}"),
        ('macro-AUC anatomy', lambda r: f"{r['macro_auc_anatomy']:.4f}"),
        ('macro-AUC ABNORMAL', lambda r: f"{r['macro_auc_abnormal']:.4f}"),
        ('macro-AUC normal', lambda r: f"{r['macro_auc_normal']:.4f}"),
        ('participation ratio', lambda r: f"{r['PR']:.2f}"),
        ('effective rank', lambda r: f"{r['eff_rank']:.2f}"),
        ('top-10 eig share', lambda r: f"{r['top10_share']:.3f}"),
    ]
    for name, fn in rows:
        print(f"{name:24s}" + "".join(f"{fn(r):>16s}" for r in results))
    print("=" * 74)


# ----------------------------- smoke -----------------------------------------
def smoke():
    print("[smoke] metric helpers ...")
    # perfect ranking -> AUC 1, AP 1
    assert abs(auc_score(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-9
    assert abs(ap_score(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-9
    # reversed -> AUC 0
    assert abs(auc_score(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) - 0.0) < 1e-9
    # ties -> 0.5
    assert abs(auc_score(np.array([0, 1]), np.array([0.5, 0.5])) - 0.5) < 1e-9

    print("[smoke] probe on synthetic (separable / random / rank-structured) ...")
    rng = np.random.default_rng(0)
    N, D = 400, 16
    Xtr = rng.standard_normal((N, D))
    Xva = rng.standard_normal((120, D))
    # finding 0: linearly separable from dim 0; finding 1: pure noise
    w = np.zeros(D); w[0] = 4.0
    def mk(X):
        p_sep = 1 / (1 + np.exp(-(X @ w)))
        y_sep = (p_sep > 0.5).astype(float)
        y_rand = (rng.random(X.shape[0]) > 0.5).astype(float)
        return np.stack([y_sep, y_rand], 1)
    Ytr, Yva = mk(Xtr), mk(Xva)
    Xtr_s, Xva_s = standardize(Xtr, Xva)
    W, b = fit_logreg(Xtr_s, Ytr)
    S = Xva_s @ W + b
    auc_sep = auc_score(Yva[:, 0], S[:, 0])
    auc_rand = auc_score(Yva[:, 1], S[:, 1])
    print(f"  separable finding AUC = {auc_sep:.3f} (expect >0.9)")
    print(f"  random    finding AUC = {auc_rand:.3f} (expect ~0.5)")
    assert auc_sep > 0.9, auc_sep
    assert 0.3 < auc_rand < 0.7, auc_rand

    print("[smoke] effective rank sanity ...")
    # isotropic -> eff_rank near D
    Ciso = rng.standard_normal((2000, D))
    lam = np.clip(np.linalg.eigvalsh(np.cov(Ciso.T)), 0, None); lam = lam[lam > 1e-12]
    er_iso = np.exp(-((lam / lam.sum()) * np.log(lam / lam.sum())).sum())
    # rank-1 -> eff_rank near 1
    v = rng.standard_normal((2000, 1)) @ rng.standard_normal((1, D))
    lam2 = np.clip(np.linalg.eigvalsh(np.cov(v.T)), 0, None); lam2 = lam2[lam2 > 1e-9]
    er_1 = np.exp(-((lam2 / lam2.sum()) * np.log(lam2 / lam2.sum())).sum())
    print(f"  isotropic eff_rank = {er_iso:.2f} (expect near {D})")
    print(f"  rank-1    eff_rank = {er_1:.2f} (expect near 1)")
    assert er_iso > 0.6 * D
    assert er_1 < 2.0
    print("[smoke] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ae_train'); ap.add_argument('--ae_val')
    ap.add_argument('--rn_train'); ap.add_argument('--rn_val')
    ap.add_argument('--label', default='negaware', choices=['raw', 'negaware'])
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    results = []
    if args.ae_train and args.ae_val:
        results.append(probe(args.ae_train, args.ae_val, args.label))
    if args.rn_train and args.rn_val:
        results.append(probe(args.rn_train, args.rn_val, args.label))
    if len(results) >= 2:
        compare(results, args.label)
    elif not results:
        ap.error('provide --ae_train/--ae_val and/or --rn_train/--rn_val, or --smoke')


if __name__ == '__main__':
    main()
