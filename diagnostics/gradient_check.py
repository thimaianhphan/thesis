"""
Phase B0 — Gradient-reachability check for the Stage-1 KG classification path.

Confirms that the loss on the KG classification head actually propagates gradient
into the visual encoder. If it doesn't (the suspected _cached_fc_feats staleness
bug), any encoder retraining is a silent no-op: the head still updates and the
loss goes down, but the encoder never learns.

Two synthetic scenarios (CPU, no real weights/images):

  A. FRESH / Stage-1 style: no cached fc_feats -> classify recomputes in-graph.
     Expect: encoder.first_conv.weight.grad is non-zero.  (This is the executed
     Stage-1 path in main_train_kg.py.)

  B. STALE / detached cache: a prior no_grad (validation) forward left a DETACHED
     fc_feats in the cache. We show the OLD read-cache-unconditionally logic
     produces ZERO encoder gradient (the invisible bug), while the NEW guarded
     logic (models/r2gen_kg.py) recomputes and restores gradient flow.

Also (optional, if constructible) exercises the REAL R2GenKGModel.classify_kg_nodes
with a random-init resnet50 (pretrained=False, no download, no AE ckpt needed) so
the fix is verified on the actual code path, not just a stand-in.

Run (CPU): python diagnostics/gradient_check.py
"""

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from modules.autoencoder import ConvEncoder


def _grad_norm(t):
    return None if t.grad is None else float(t.grad.abs().sum())


def standin():
    """Reproduce the cache decision (old vs new) over a real ConvEncoder + head."""
    print("=" * 70)
    print("B0 stand-in: ConvEncoder -> GAP -> Linear head (synthetic)")
    print("=" * 70)
    torch.manual_seed(0)
    enc = ConvEncoder().train()
    head = nn.Linear(256, 5)
    x = torch.randn(2, 3, 224, 224)
    y = torch.zeros(2, 5); y[0, 1] = 1.0; y[1, 3] = 1.0
    first_conv = enc.net[0].weight

    def classify_OLD(cache):
        # old logic: use cache whenever it's not None
        if cache is not None:
            fc = cache
        else:
            fc = enc(x).mean(dim=(2, 3))
        return head(fc)

    def classify_NEW(cache):
        # new logic: only trust cache if connected to current graph
        usable = cache is not None and (cache.requires_grad or not torch.is_grad_enabled())
        if usable:
            fc = cache
        else:
            fc = enc(x).mean(dim=(2, 3))
        return head(fc)

    # ---- Scenario A: fresh (no cache) ----
    enc.zero_grad(); head.zero_grad()
    logits = classify_NEW(None)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    gA = _grad_norm(first_conv)
    print(f"[A fresh/Stage-1]      encoder first-conv grad = {gA}")
    assert gA is not None and gA > 0, "FAIL: fresh path gives no encoder gradient"

    # ---- Scenario B: stale detached cache (from a no_grad validation forward) ----
    with torch.no_grad():
        stale = enc(x).mean(dim=(2, 3))     # detached: requires_grad == False
    assert not stale.requires_grad

    enc.zero_grad(); head.zero_grad()
    logits = classify_OLD(stale)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    gB_old = _grad_norm(first_conv)
    print(f"[B stale] OLD logic    encoder first-conv grad = {gB_old}   <-- silent no-op bug")
    assert gB_old is None or gB_old == 0, "expected OLD logic to starve encoder"

    enc.zero_grad(); head.zero_grad()
    logits = classify_NEW(stale)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    gB_new = _grad_norm(first_conv)
    print(f"[B stale] NEW logic    encoder first-conv grad = {gB_new}   <-- fixed (guard recomputes)")
    assert gB_new is not None and gB_new > 0, "FAIL: guard did not restore gradient"
    print("stand-in: PASS (bug reproduced with OLD logic, fixed with NEW guard)\n")


def real_model():
    """Exercise the REAL fixed classify_kg_nodes on a minimal model (optional)."""
    print("=" * 70)
    print("B0 real path: R2GenKGModel.classify_kg_nodes (resnet50, random init)")
    print("=" * 70)
    ann = os.path.join(_ROOT, 'data', 'iu_xray', 'annotation.json')
    if not os.path.exists(ann):
        print(f"[skip] annotation not found at {ann}")
        return
    try:
        from types import SimpleNamespace
        from modules.tokenizers import Tokenizer
        from models.r2gen_kg import R2GenKGModel
        args = SimpleNamespace(
            image_dir='data/iu_xray/images/', ann_path=ann, dataset_name='iu_xray',
            max_seq_length=60, threshold=3, num_workers=0, batch_size=2,
            visual_extractor='resnet50', visual_extractor_pretrained=False, d_vf=2048,
            d_model=512, d_ff=512, num_heads=8, num_layers=3, dropout=0.1,
            logit_layers=1, bos_idx=0, eos_idx=0, pad_idx=0, use_bn=0, drop_prob_lm=0.5,
            rm_num_slots=3, rm_num_heads=8, rm_d_model=512,
            kg_num_gcn_layers=1, kg_gcn_alpha=0.2, kg_loss_weight=0.1,
            kg_co_occur_threshold=3, use_expert_memory=False,
            expert_query_causal_mean=False,
        )
        tok = Tokenizer(args)
        model = R2GenKGModel(args, tok).train()
    except Exception as e:
        print(f"[skip] could not build real model on CPU: {type(e).__name__}: {e}")
        return

    images = torch.randn(2, 2, 3, 224, 224)  # iu_xray: [B, 2 views, 3, H, W]
    reports = ['no pleural effusion. lungs clear.', 'large effusion and cardiomegaly.']
    kg_labels = model.encoder_decoder.get_kg_labels(reports)

    # Simulate a prior no_grad validation forward that pollutes the cache.
    model.eval()
    with torch.no_grad():
        model(images, mode='sample')
    assert getattr(model, '_cached_fc_feats', None) is not None
    assert not model._cached_fc_feats.requires_grad, "validation cache should be detached"

    # Now Stage-1 style: train mode, classify WITHOUT a fresh forward first.
    model.train()
    model.zero_grad()
    logits = model.classify_kg_nodes(images)
    loss = model.kg_classifier.get_loss(logits, kg_labels)
    loss.backward()

    # find the encoder's first conv weight
    first = None
    for n, p in model.visual_extractor.named_parameters():
        if p.ndim == 4:
            first = (n, p)
            break
    g = _grad_norm(first[1]) if first else None
    print(f"[real, polluted cache] logits shape = {tuple(logits.shape)}")
    print(f"[real, polluted cache] encoder '{first[0]}' grad = {g}")
    assert g is not None and g > 0, "FAIL: real classify_kg_nodes starves encoder"
    print("real path: PASS (guarded classify_kg_nodes propagates gradient)\n")


if __name__ == '__main__':
    standin()
    real_model()
    print("B0 gradient-reachability check: DONE")
