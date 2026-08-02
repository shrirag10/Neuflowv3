"""Pre-flight checks for the training and eval path. Runs on CPU in ~2 min.

Run this before spending cluster hours. Every check is a property that must
hold for the results to mean anything; each one below failed at some point
during development, which is why it is here.

    python3 scripts/verify_pipeline.py
"""

import sys, os, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from NeuFlow.neuflow import NeuFlow
from utils.load_model import (my_load_weights, load_with_new_keys, my_freeze_model,
                              set_frozen_bn_eval)

DEV = torch.device('cuda' if (os.environ.get('VERIFY_GPU') and torch.cuda.is_available())
                   else 'cpu')
AMP = DEV.type == 'cuda'
H, W = 128, 256
PASS, FAIL = [], []


def check(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print(f'[{"PASS" if ok else "FAIL"}] {name}' + (f'  --  {detail}' if detail else ''))


def build(head='convex', unc=False, freeze=True):
    m = NeuFlow(use_implicit=True, head_mode=head, predict_uncertainty=unc).to(DEV)
    load_with_new_keys(m, my_load_weights('neuflow_mixed.pth'),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
    if freeze:
        my_freeze_model(m)
    m.eval()
    m.init_bhwd(1, H, W, DEV, amp=AMP)
    return m


def real_pair():
    """A real image pair if one is on disk, else None.

    Some properties (notably the stride-2 dense approximation) depend on flow
    being spatially smooth, which is true of real imagery and false of random
    noise. Testing those on noise gives a misleading failure.
    """
    import glob
    for pat in ('datasets/vkitti2/Scene18/clone/frames/rgb/Camera_0/rgb_%05d.jpg',
                'test_images/*.jpg'):
        if '%' in pat:
            f1, f2 = pat % 100, pat % 101
            if os.path.exists(f1) and os.path.exists(f2):
                pair = [f1, f2]
            else:
                continue
        else:
            g = sorted(glob.glob(pat))
            if len(g) < 2:
                continue
            pair = g[:2]
        import cv2
        ims = []
        for f in pair:
            im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)[:H, :W]
            if im.shape[:2] != (H, W):
                im = cv2.resize(im, (W, H))
            t = torch.from_numpy(im).permute(2, 0, 1).float()[None].to(DEV)
            ims.append(t.half() if AMP else t)
        return ims[0], ims[1], os.path.basename(pair[0])
    return None


def main():
    print(f'device: {DEV}  amp/fp16: {AMP}\n')
    torch.manual_seed(0)
    a = (torch.rand(1, 3, H, W) * 255).to(DEV)
    b = (torch.rand(1, 3, H, W) * 255).to(DEV)
    if AMP:
        a, b = a.half(), b.half()

    # ---- 1. BatchNorm really is frozen in train mode -----------------------
    m = build()
    snap = {k: v.clone() for k, v in m.state_dict().items()
            if k.endswith(('.running_mean', '.running_var', '.num_batches_tracked'))}
    m.train()
    n_bn = set_frozen_bn_eval(m)
    with torch.amp.autocast('cuda', enabled=AMP):
        for _ in range(2):
            m(a, b, iters_s16=1, iters_s8=2)
    now = m.state_dict()
    drift = max((now[k].float() - snap[k].float()).abs().max().item() for k in snap)
    check('BatchNorm frozen during training', n_bn > 0 and drift == 0.0,
          f'{n_bn} BN layers held, max drift {drift}')

    # ---- 2. only the decoder is trainable ----------------------------------
    m = build()
    tr = {n for n, p in m.named_parameters() if p.requires_grad}
    leaked = {n for n in tr if not n.startswith('implicit_decoder_module')}
    check('only decoder params trainable', len(tr) > 0 and not leaked,
          f'{len(tr)} trainable tensors, {len(leaked)} outside the decoder')

    # ---- 3. untrained convex head == bilinear upsampling -------------------
    m = build()
    hd = m.implicit_decoder_module.convex_head
    torch.nn.init.zeros_(hd.layers[-1].weight); torch.nn.init.zeros_(hd.layers[-1].bias)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=AMP):
        st = m.infer_coarse_state(a, b)
        dense = m.decode_dense_fast(st, stride=1)
        bilin = torch.nn.functional.interpolate(
            st['coarse_flow_s8'].float(), scale_factor=8, mode='bilinear',
            align_corners=False) * 8
    d = (dense - bilin).abs().max().item()
    check('zero-init decoder reproduces bilinear', d < 0.05, f'max deviation {d:.4f} px')

    # ---- 4. sparse queries agree with the dense field ----------------------
    m = build()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=AMP):
        st = m.infer_coarse_state(a, b)
        dense = m.decode_dense_fast(st, stride=1)
        ys = torch.randint(0, H, (64,)); xs = torch.randint(0, W, (64,))
        q = torch.stack([xs.float(), ys.float()], -1)[None].to(DEV)
        sp = m.decode_queries(st, query_coords=q)[0]
    ref = dense[0, :, ys.to(DEV), xs.to(DEV)].T
    d = (sp - ref).abs().max().item()
    check('sparse query == dense at same coords', d < 1e-3, f'max diff {d:.6f} px')

    # ---- 5. uncertainty works in BOTH decode paths -------------------------
    m = build(unc=True)
    ok, detail = True, ''
    try:
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=AMP):
            st = m.infer_coarse_state(a, b)
            q = torch.tensor([[[10.0, 10.0], [50.5, 33.2]]], device=DEV)
            fs, bs = m.decode_queries(st, query_coords=q, return_uncertainty=True)
            fd, bd = m.decode_dense_fast(st, stride=2, return_uncertainty=True)
        ok = (fs.shape[-1] == 2 and bs.shape[-1] == q.shape[1]
              and fd.shape[1] == 2 and bd.shape[1] == 1 and bd.shape[-2:] == fd.shape[-2:])
        detail = (f'sparse flow {tuple(fs.shape)} b {tuple(bs.shape)} | '
                  f'dense flow {tuple(fd.shape)} b {tuple(bd.shape)}')
    except Exception as e:
        ok, detail = False, f'{type(e).__name__}: {e}'
    check('uncertainty head works in sparse AND fast-dense', ok, detail)

    # ---- 6. predicted uncertainty is positive and finite -------------------
    if ok:
        bvals = bd[0, 0]
        check('predicted uncertainty positive and finite',
              bool(torch.isfinite(bvals).all() and (bvals > 0).all()),
              f'b range [{bvals.min():.4f}, {bvals.max():.4f}]')

    # ---- 7. stride-2 dense close to stride-1 (real imagery only) -----------
    rp = real_pair()
    if rp is None:
        print('[SKIP] stride-2 check: no real image pair on disk')
    else:
        ra, rb, tag = rp
        m = build()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=AMP):
            st = m.infer_coarse_state(ra, rb)
            d1 = m.decode_dense_fast(st, stride=1)
            d2 = m.decode_dense_fast(st, stride=2)
        diff = (d1 - d2).abs().mean().item()
        check('stride-2 dense approximates stride-1 (real imagery)', diff < 0.05,
              f'mean diff {diff:.4f} px on {tag}')

    # ---- 8. training is reproducible under a fixed seed --------------------
    def short_run(seed):
        out = subprocess.run(
            [sys.executable, 'train.py', '--stage', 'FlyingChairs', '--implicit',
             '--sparse_loss', '--head', 'convex', '--num_sparse_points', '128',
             '--batch_size', '2', '--num_workers', '0', '--seed', str(seed),
             '--lr', '2e-4', '--onecycle', '--num_steps', '2', '--val_freq', '999999',
             '--val_dataset', 'none', '--resume', 'neuflow_mixed.pth',
             '--checkpoint_dir', tempfile.mkdtemp()],
            capture_output=True, text=True, timeout=900)
        eps = [ln.split('epe=')[-1].split(',')[0].split(']')[0]
               for ln in out.stderr.split('\r') if 'epe=' in ln]
        return eps[-1] if eps else None

    r1, r2, r3 = short_run(1234), short_run(1234), short_run(999)
    check('same seed gives identical training', r1 is not None and r1 == r2,
          f'seed1234 runs: {r1} vs {r2}')
    check('different seed gives different training', r1 != r3,
          f'seed1234 {r1} vs seed999 {r3}')

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED:', ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
