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


def effective_rank_stats(X):
    """Mean-center X [N,D]; return (eff_rank, participation_ratio, top10_share).
    Shared by probe() and train_ae_combined.py's per-epoch rank logger (Task 2)
    so both use the exact same metric definition -- no drift between the two."""
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    lam = np.clip(np.linalg.eigvalsh(cov), 0, None)
    lam = lam[lam > 1e-12]
    if len(lam) == 0:
        return 0.0, 0.0, 0.0
    p = lam / lam.sum()
    PR = float((lam.sum() ** 2) / (lam ** 2).sum())
    H = float(-(p * np.log(p)).sum())
    eff_rank = float(np.exp(H))
    top10 = float(np.sort(lam)[::-1][:10].sum() / lam.sum())
    return eff_rank, PR, top10


def probe(train_npz, val_npz, label='negaware', xkey='X', verbose=True):
    tr = np.load(train_npz, allow_pickle=True)
    va = np.load(val_npz, allow_pickle=True)
    ykey = 'y_negaware' if label == 'negaware' else 'y_raw'
    if xkey not in tr.files or xkey not in va.files:
        raise SystemExit(f"[error] '{xkey}' not found in {train_npz if xkey not in tr.files else val_npz} "
                          f"(available: {tr.files}). Was it extracted with the Task-1 spatial-pooling "
                          f"extract_features.py, with --encoder ae?")
    Xtr, Ytr = tr[xkey].astype(np.float64), tr[ykey].astype(np.float64)
    Xva, Yva = va[xkey].astype(np.float64), va[ykey].astype(np.float64)
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
    eff_rank, PR, top10 = effective_rank_stats(Xtr)

    res = {
        'encoder': str(tr['encoder']), 'D': D, 'label': label, 'pool': xkey,
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


def _attn_forward_backward(Z, Y, q, W, b, l2=1e-2):
    """One forward + manual-backward step of the single-query attention pool.
    Z: [N,P,D] standardized spatial map. q:[D], W:[D,C], b:[C]. Y:[N,C].
    Pool: alpha = softmax(Z@q, axis=1); pooled = sum_p alpha*Z; logits = pooled@W+b.
    Returns (logits, loss, dq, dW, db) -- all grads are already /N (mean-loss scale),
    matching fit_logreg's convention so the same lr/iters behave similarly."""
    N = Z.shape[0]
    scores = np.einsum('npd,d->np', Z, q)                      # [N,P]
    scores = scores - scores.max(axis=1, keepdims=True)         # softmax stability
    ex = np.exp(scores)
    alpha = ex / ex.sum(axis=1, keepdims=True)                  # [N,P]
    pooled = np.einsum('np,npd->nd', alpha, Z)                  # [N,D]
    logits = pooled @ W + b                                     # [N,C]
    Phat = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    eps = 1e-9
    loss = -(Y * np.log(Phat + eps) + (1 - Y) * np.log(1 - Phat + eps)).mean()

    dlogits = (Phat - Y) / N                                    # [N,C]
    dW = pooled.T @ dlogits + l2 * W                            # [D,C]
    db = dlogits.sum(axis=0)                                    # [C]
    dpooled = dlogits @ W.T                                     # [N,D]
    dalpha = np.einsum('nd,npd->np', dpooled, Z)                # [N,P]
    s = np.einsum('np,np->n', alpha, dalpha)                    # [N]
    dscores = alpha * (dalpha - s[:, None])                     # [N,P] softmax jacobian-vector product
    dq = np.einsum('np,npd->d', dscores, Z)                     # [D]
    return logits, float(loss), dq, dW, db


def fit_attn_probe(Ztr, Ytr, l2=1e-2, lr=0.5, iters=300, seed=0, log_every=50):
    """Full-batch gradient descent for the single-query attention pool+head.
    W is small-random (not zero) at init: dpooled = dlogits @ W.T, so a
    zero-initialized W would make the very first step's gradient into q exactly
    zero (the pooling weights can't learn until W has moved at least once)."""
    rng = np.random.default_rng(seed)
    N, P, D = Ztr.shape
    C = Ytr.shape[1]
    q = rng.standard_normal(D) * 0.01
    W = rng.standard_normal((D, C)) * 0.01
    b = np.zeros(C)
    for it in range(iters):
        _, loss, dq, dW, db = _attn_forward_backward(Ztr, Ytr, q, W, b, l2=l2)
        q -= lr * dq; W -= lr * dW; b -= lr * db
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"    [attn-probe] iter {it:4d}/{iters}  loss={loss:.4f}")
    return q, W, b


def _load_spatial_standardized(train_npz, val_npz):
    """Load map_f16 [N,D,H,W] from both files, reshape to [N,P,D], standardize
    per-channel on TRAIN stats (flattened over N*P). float32 to bound memory."""
    tr = np.load(train_npz, allow_pickle=True)
    va = np.load(val_npz, allow_pickle=True)
    if 'map_f16' not in tr.files or 'map_f16' not in va.files:
        raise SystemExit("[error] --pool attn requires 'map_f16' in both npz files "
                          "-- re-run extract_features.py with --save_spatial.")

    def reshape(m):
        m = m.astype(np.float32)                    # [N,D,H,W]
        Nn, D, H, W = m.shape
        return m.reshape(Nn, D, H * W).transpose(0, 2, 1)  # [N,P,D]

    Ztr = reshape(tr['map_f16'])
    Zva = reshape(va['map_f16'])
    mu = Ztr.reshape(-1, Ztr.shape[-1]).mean(0)
    sd = Ztr.reshape(-1, Ztr.shape[-1]).std(0) + 1e-6
    Ztr = (Ztr - mu) / sd
    Zva = (Zva - mu) / sd
    return Ztr, Zva, tr, va


def probe_attn(train_npz, val_npz, label='negaware', l2=1e-2, lr=0.5, iters=300, verbose=True):
    """Attention-probe: single learned query + linear head, jointly fit (gradient
    descent, numpy-only). Bounded beyond pure-linear (one query vector + one
    linear layer), no encoder training. Needs --save_spatial features."""
    Ztr, Zva, tr, va = _load_spatial_standardized(train_npz, val_npz)
    ykey = 'y_negaware' if label == 'negaware' else 'y_raw'
    Ytr = tr[ykey].astype(np.float64)
    Yva = va[ykey].astype(np.float64)
    node_list = list(tr['node_list'])
    node_types = list(tr['node_types'])
    D = Ztr.shape[2]

    print(f"    [attn-probe] fitting on Z={Ztr.shape} (this may take a while, pure numpy)...")
    q, W, b = fit_attn_probe(Ztr.astype(np.float64), Ytr, l2=l2, lr=lr, iters=iters)

    # val forward only (no label use)
    scores = np.einsum('npd,d->np', Zva.astype(np.float64), q)
    scores = scores - scores.max(axis=1, keepdims=True)
    ex = np.exp(scores)
    alpha = ex / ex.sum(axis=1, keepdims=True)
    pooled_va = np.einsum('np,npd->nd', alpha, Zva.astype(np.float64))
    S = pooled_va @ W + b

    pooled_tr_scores = np.einsum('npd,d->np', Ztr.astype(np.float64), q)
    pooled_tr_scores -= pooled_tr_scores.max(axis=1, keepdims=True)
    ex_tr = np.exp(pooled_tr_scores)
    alpha_tr = ex_tr / ex_tr.sum(axis=1, keepdims=True)
    pooled_tr = np.einsum('np,npd->nd', alpha_tr, Ztr.astype(np.float64))

    per = []
    for c in range(Ytr.shape[1]):
        per.append({
            'name': node_list[c], 'type': node_types[c],
            'n_train_pos': int(Ytr[:, c].sum()), 'n_val_pos': int(Yva[:, c].sum()),
            'auc': auc_score(Yva[:, c], S[:, c]),
            'ap': ap_score(Yva[:, c], S[:, c]),
        })

    def macro(types=None):
        v = [p['auc'] for p in per if not np.isnan(p['auc']) and (types is None or p['type'] in types)]
        return float(np.mean(v)) if v else np.nan

    mask = np.array([not np.isnan(p['auc']) for p in per])
    micro = auc_score(Yva[:, mask].ravel(), S[:, mask].ravel()) if mask.any() else np.nan
    eff_rank, PR, top10 = effective_rank_stats(pooled_tr)

    res = {
        'encoder': f"{str(tr['encoder'])}[attn]", 'D': D, 'label': label, 'pool': 'attn',
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
    print(f"AE vs ResNet - comparison table (labels = {label})")
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


POOL_XKEY = {'gap': 'X', 'gmp': 'X_gmp', 'lse': 'X_lse'}


def run_pool_mode(train_npz, val_npz, pool, label, attn_iters=300):
    """Task 1: single-file spatial-pool comparison. Always runs the gap baseline
    (the existing X, i.e. the value already reported as e.g. 0.628) plus the
    requested --pool, and prints the delta so the decision-table thresholds in
    the Task 1 spec can be read directly off this output."""
    tr = np.load(train_npz, allow_pickle=True)
    encoder = str(tr['encoder'])
    print(f"[Task 1] file={train_npz}  encoder={encoder}  requested pool={pool}  labels={label}")

    base = probe(train_npz, val_npz, label=label, xkey='X', verbose=False)
    print("\n--- baseline: gap (reproduces the historical GAP number) ---")
    _print_one(base)

    if pool == 'gap':
        print("\n[Task 1] --pool gap requested == baseline; nothing further to compare.")
        return base, base

    if pool == 'attn':
        req = probe_attn(train_npz, val_npz, label=label, iters=attn_iters)
    else:
        xkey = POOL_XKEY[pool]
        req = probe(train_npz, val_npz, label=label, xkey=xkey, verbose=False)
        req['encoder'] = f"{encoder}[{pool}]"

    print(f"\n--- requested: {pool} ---")
    _print_one(req)

    delta = req['macro_auc_abnormal'] - base['macro_auc_abnormal']
    print("\n" + "=" * 60)
    print(f"[Task 1] abnormal macro-AUC   gap={base['macro_auc_abnormal']:.4f}   "
          f"{pool}={req['macro_auc_abnormal']:.4f}   delta={delta:+.4f}")
    if delta >= 0.03:
        print("  >= +0.03 vs gap: consider this evidence pooling was hiding signal.")
    elif delta <= -0.01:
        print(f"  {pool} <= gap: consistent with 'gmp/lse < gap' row (diffuse, not peaky, signal).")
    else:
        print("  ~flat vs gap: pooling is not where the information is.")
    print("=" * 60)
    return base, req


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

    print("[smoke] effective_rank_stats() shared-function refactor matches inline calc ...")
    er2, pr2, top10_2 = effective_rank_stats(Ciso)
    assert abs(er2 - er_iso) < 1e-6, (er2, er_iso)
    print(f"  effective_rank_stats(isotropic) = {er2:.2f}  matches inline computation  OK")

    print("[smoke] attention-probe (Task 1) forward+backward on synthetic ...")
    rng2 = np.random.default_rng(1)
    Nb, P, Db, Cn = 8, 196, 256, 74
    Zsyn = rng2.standard_normal((Nb, P, Db))
    Ysyn = (rng2.random((Nb, Cn)) > 0.7).astype(np.float64)
    q0 = rng2.standard_normal(Db) * 0.01
    W0 = rng2.standard_normal((Db, Cn)) * 0.01  # NOT zero: dpooled=dlogits@W.T needs W!=0 to reach q
    b0 = np.zeros(Cn)
    logits, loss0, dq, dW, db = _attn_forward_backward(Zsyn, Ysyn, q0, W0, b0)
    assert logits.shape == (Nb, Cn), logits.shape
    assert np.isfinite(loss0) and loss0 > 0
    assert np.abs(dq).sum() > 0, "query vector got no gradient"
    assert np.abs(dW).sum() > 0, "head weight got no gradient"
    print(f"  logits {tuple(logits.shape)}  loss={loss0:.4f}  "
          f"|dq|_1={np.abs(dq).sum():.4f}  |dW|_1={np.abs(dW).sum():.4f}  (both non-zero)")
    # a few fit steps should not explode and should not increase loss vs the first step
    q1, W1, b1 = fit_attn_probe(Zsyn, Ysyn, iters=20, log_every=0)
    _, loss1, _, _, _ = _attn_forward_backward(Zsyn, Ysyn, q1, W1, b1)
    assert np.isfinite(loss1)
    assert loss1 <= loss0 + 1e-6, f"loss should not increase over 20 steps: {loss0} -> {loss1}"
    print(f"  20-step fit: loss {loss0:.4f} -> {loss1:.4f}  (non-increasing)  OK")

    print("[smoke] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('train_npz', nargs='?', default=None,
                    help='Task 1 mode: train npz (paired with val_npz below) to compare --pool against gap.')
    ap.add_argument('val_npz', nargs='?', default=None, help='Task 1 mode: val npz.')
    ap.add_argument('--pool', default='gap', choices=['gap', 'gmp', 'lse', 'attn'],
                    help='Task 1 mode only: which pooling to compare against the gap baseline.')
    ap.add_argument('--attn_iters', type=int, default=300, help='gradient steps for --pool attn.')
    ap.add_argument('--ae_train'); ap.add_argument('--ae_val')
    ap.add_argument('--rn_train'); ap.add_argument('--rn_val')
    ap.add_argument('--label', default='negaware', choices=['raw', 'negaware'])
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return

    if args.train_npz and args.val_npz:
        run_pool_mode(args.train_npz, args.val_npz, args.pool, args.label, args.attn_iters)
        return

    results = []
    if args.ae_train and args.ae_val:
        results.append(probe(args.ae_train, args.ae_val, args.label))
    if args.rn_train and args.rn_val:
        results.append(probe(args.rn_train, args.rn_val, args.label))
    if len(results) >= 2:
        compare(results, args.label)
    elif not results:
        ap.error('provide train_npz val_npz (+ --pool), or --ae_train/--ae_val and/or '
                 '--rn_train/--rn_val, or --smoke')


if __name__ == '__main__':
    main()
