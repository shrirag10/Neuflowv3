"""End-to-end video-pipeline FPS: v2 vs v3 modes on identical frames.

Grabs frames from a video source (YouTube URL, file, or synthetic), then times
each pipeline mode over the same frame pairs:

    v2 dense          full-frame flow map (v2's only mode)
    v3 motion         coarse pass + ego-compensated motion boxes (GUI playback mode)
    v3 sparse-800     coarse pass + 800 point queries
    v3 dense          coarse pass + all-pixel decode (evaluation mode)

Usage:
    python3 scripts/benchmark_fps.py [--youtube URL | --video file] [--frames 60]
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import torch

from NeuFlow.neuflow import NeuFlow
from data_utils import frame_utils
from utils.load_model import my_load_weights, load_with_new_keys

DEVICE = torch.device('cuda')


def grab_frames(args):
    src = args.video
    if args.youtube:
        import yt_dlp
        with yt_dlp.YoutubeDL({'format': 'best[ext=mp4][height<=720]', 'quiet': True}) as y:
            src = y.extract_info(args.youtube, download=False)['url']
    frames = []
    if src:
        cap = cv2.VideoCapture(src)
        while len(frames) < args.frames:
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[:2]
            if w > 1024:
                f = cv2.resize(f, (1024, int(h * 1024 / w)))
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
    if not frames:
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (576, 1024, 3), dtype=np.uint8) for _ in range(args.frames)]
        print('using synthetic frames')
    print(f'{len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}')
    return frames


def load(implicit, head='convex', ckpt=None):
    m = NeuFlow(use_implicit=implicit, head_mode=head).to(DEVICE)
    load_with_new_keys(m, my_load_weights(ckpt or 'neuflow_mixed.pth'),
                       missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                       unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
    m.eval()
    return m


def to_tensor(f):
    return torch.from_numpy(f).permute(2, 0, 1).float()[None]


def bench(name, frames, fn, warmup=3):
    for k in range(warmup):
        fn(frames[k], frames[k + 1])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for a, b in zip(frames[:-1], frames[1:]):
        fn(a, b)
        n += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f'{name:<46} {n / dt:6.1f} FPS   ({dt / n * 1000:6.1f} ms/pair)')
    return n / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--youtube'); ap.add_argument('--video')
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--checkpoint', default='checkpoints/neuflowv3_mix/step_030000.pth')
    args = ap.parse_args()

    frames = grab_frames(args)
    padder = frame_utils.InputPadder(to_tensor(frames[0]).shape, padding_factor=16)
    H, W = padder.pad(to_tensor(frames[0]).to(DEVICE))[0].shape[-2:]

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from query_gui import detect_motion

    results = {}

    v2 = load(False)
    v2.init_bhwd(1, H, W, DEVICE)
    def run_v2(a, b):
        x, y = padder.pad(to_tensor(a).to(DEVICE), to_tensor(b).to(DEVICE))
        with torch.no_grad(), torch.amp.autocast('cuda'):
            v2(x, y)
    results['v2 dense (only mode)'] = bench('NeuFlow v2 — dense map (its only mode)', frames, run_v2)
    del v2; torch.cuda.empty_cache()

    v3 = load(True, ckpt=args.checkpoint)
    v3.init_bhwd(1, H, W, DEVICE)

    def run_motion(a, b):
        x, y = padder.pad(to_tensor(a).to(DEVICE), to_tensor(b).to(DEVICE))
        with torch.no_grad(), torch.amp.autocast('cuda'):
            state = v3.infer_coarse_state(x, y)
        detect_motion(state)
    results['v3 motion detection'] = bench('NeuFlow v3 — coarse + motion boxes (GUI mode)', frames, run_motion)

    q = torch.rand(1, 800, 2, device=DEVICE) * torch.tensor([W - 1., H - 1.], device=DEVICE)
    def run_sparse(a, b):
        x, y = padder.pad(to_tensor(a).to(DEVICE), to_tensor(b).to(DEVICE))
        with torch.no_grad(), torch.amp.autocast('cuda'):
            state = v3.infer_coarse_state(x, y)
            v3.decode_queries(state, query_coords=q)
    results['v3 sparse 800'] = bench('NeuFlow v3 — coarse + 800 point queries', frames, run_sparse)

    def run_dense3(a, b):
        x, y = padder.pad(to_tensor(a).to(DEVICE), to_tensor(b).to(DEVICE))
        with torch.no_grad(), torch.amp.autocast('cuda'):
            v3(x, y)
    results['v3 dense (eval mode)'] = bench('NeuFlow v3 — dense all-pixel decode (eval only)', frames, run_dense3)

    print('\nSummary: at matched resolution, v3 answers motion/sparse questions at '
          f'{results["v3 motion detection"] / results["v2 dense (only mode)"]:.2f}x '
          'the FPS of v2 producing its full map.')


if __name__ == '__main__':
    main()
