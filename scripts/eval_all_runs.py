"""Evaluate every finished training run on VKITTI2 Scene18+20 and print one table.

Skips runs whose checkpoint is not there yet, so it is safe to run while other
jobs are still training. Handles the uncertainty run's extra flag automatically.

    python3 scripts/eval_all_runs.py                      # final checkpoints
    python3 scripts/eval_all_runs.py --step 50000         # a specific step
    python3 scripts/eval_all_runs.py --fast_dense --stride 2
"""

import sys, os, re, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

RUNS = [
    ('v3_FlyingChairs',                            'FlyingChairs',                  False),
    ('v3_FlyingChairs_VKITTI2',                    'FlyingChairs+VKITTI2',          False),
    ('v3_FlyingChairs_VKITTI2_Sintel',             'FlyingChairs+VKITTI2+Sintel',   False),
    ('v3_FlyingChairs_VKITTI2_Sintel_uncertainty', 'above + uncertainty head',      True),
    ('v3_FlyingChairs_VKITTI2_Sintel_Spring',      '+Spring',                       False),
]


def run_eval(ckpt, data_root, uncertainty, extra):
    cmd = [sys.executable, os.path.join(HERE, 'eval_vkitti2.py'),
           '--head', 'convex', '--checkpoint', ckpt, '--dataset_root', data_root]
    if uncertainty:
        cmd.append('--uncertainty')
    cmd += extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ['(no stderr)'])[-1]
    out = p.stdout
    def grab(pat):
        m = re.search(pat, out)
        return float(m.group(1)) if m else float('nan')
    return {
        'epe':  grab(r'Mean EPE\s*:\s*([\d.]+)'),
        'a1':   grab(r'1px acc\s*:\s*([\d.]+)'),
        'a3':   grab(r'3px acc\s*:\s*([\d.]+)'),
        'ms':   grab(r'Mean\s*:\s*([\d.]+) ms'),
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_root', default=f'/scratch/{os.environ.get("USER","")}/neuflow_ckpts')
    ap.add_argument('--dataset_root', default=f'/scratch/{os.environ.get("USER","")}/neuflow_datasets/vkitti2')
    ap.add_argument('--step', default='100000')
    ap.add_argument('--fast_dense', action='store_true')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--skip_v2', action='store_true')
    args = ap.parse_args()

    extra = []
    if args.fast_dense:
        extra += ['--fast_dense', '--stride', str(args.stride)]

    rows = []

    if not args.skip_v2:
        v2 = os.path.join(ROOT, 'neuflow_mixed.pth')
        if os.path.exists(v2):
            print('evaluating NeuFlow v2 baseline ...', flush=True)
            cmd = [sys.executable, os.path.join(HERE, 'eval_vkitti2.py'),
                   '--no_implicit', '--checkpoint', v2,
                   '--dataset_root', args.dataset_root]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0:
                g = lambda pat: float(re.search(pat, p.stdout).group(1))
                rows.append(('NeuFlow v2 (reference)', 'FlyingThings (theirs)',
                             {'epe': g(r'Mean EPE\s*:\s*([\d.]+)'),
                              'a1': g(r'1px acc\s*:\s*([\d.]+)'),
                              'a3': g(r'3px acc\s*:\s*([\d.]+)'),
                              'ms': g(r'Mean\s*:\s*([\d.]+) ms')}))
            else:
                print('  v2 eval failed:', p.stderr.strip().splitlines()[-1])

    for name, desc, unc in RUNS:
        ckpt = os.path.join(args.ckpt_root, name, f'step_{int(args.step):06d}.pth')
        if not os.path.exists(ckpt):
            print(f'skip {name}: no {os.path.basename(ckpt)} yet', flush=True)
            continue
        print(f'evaluating {name} ...', flush=True)
        res, err = run_eval(ckpt, args.dataset_root, unc, extra)
        if res is None:
            print(f'  FAILED: {err}')
            continue
        rows.append((name, desc, res))

    if not rows:
        print('\nnothing to report yet.')
        return

    mode = f'fast_dense stride {args.stride}' if args.fast_dense else 'standard dense'
    print(f'\nVKITTI2 Scene18+20, 1174 pairs, per-pixel   [{mode}]  step {args.step}')
    print(f'{"run":46s} {"EPE":>7s} {"1px%":>7s} {"3px%":>7s} {"ms":>7s}')
    print('-' * 78)
    for name, desc, r in rows:
        print(f'{name:46s} {r["epe"]:7.3f} {r["a1"]:7.2f} {r["a3"]:7.2f} {r["ms"]:7.1f}')
    print('\ntraining data per run:')
    for name, desc, _ in rows:
        print(f'  {name:46s} {desc}')
    print('\nNote: v2 was trained by its authors on FlyingThings, not on these '
          'mixes.\nOnly v3_FlyingChairs is a clean like-for-like comparison '
          '(no driving data in training).')


if __name__ == '__main__':
    main()
