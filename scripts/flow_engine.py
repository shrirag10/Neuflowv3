"""Compute core shared by the GUI and the headless benchmarks.

Kept free of any GUI dependency on purpose: the cluster has no PyQt5, and the
benchmarks need exactly this and nothing else. scripts/video_region_gui.py
imports FlowEngine from here.

Two region modes, which cost very different amounts:

  QUERY  run the full coarse pass on the whole frame (unavoidable: matching
         needs global context), then decode ONLY the pixels inside the box.
         Exact, identical to the full-frame result inside the box. Saves decode
         time, which is the small part.

  CROP   feed only the box plus a margin through the entire pipeline. Cost
         scales with area, so a quarter-size box is roughly 4x faster end to
         end. An approximation: the network loses the surrounding context that
         global matching uses, and motion leaving the crop cannot be found.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

from NeuFlow.neuflow import NeuFlow
from data_utils import frame_utils
from utils.load_model import my_load_weights, load_with_new_keys


# --------------------------------------------------------------------------- model
class FlowEngine:
    """Wraps v3 (queryable) and v2 (dense baseline) behind one interface."""

    def __init__(self, ckpt_v3, ckpt_v2='neuflow_mixed.pth', head='convex',
                 uncertainty=False):
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amp = self.dev.type == 'cuda'
        self.uncertainty = uncertainty

        self.v3 = NeuFlow(use_implicit=True, head_mode=head,
                          predict_uncertainty=uncertainty).to(self.dev)
        load_with_new_keys(self.v3, my_load_weights(ckpt_v3),
                           missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                           unexpected_ok_substrings=['conv_s8', 'upsample_s8'])
        self.v3.eval()

        self.v2 = None
        if ckpt_v2 and os.path.exists(ckpt_v2):
            self.v2 = NeuFlow(use_implicit=False).to(self.dev)
            load_with_new_keys(self.v2, my_load_weights(ckpt_v2),
                               missing_ok_substrings=['implicit_decoder_module', 'win_proj_'],
                               unexpected_ok_substrings=[])
            self.v2.eval()

        self._shape = {}
        self.state = None          # cached coarse state for the current pair
        self.state_key = None
        self.coarse_ms = 0.0
        self._warm = set()

    def warmup(self, h, w, n=3):
        """Run the pipeline a few times before timing anything.

        The first CUDA call of a given shape pays kernel autotuning and memory
        allocation -- measured here at 367 ms against a ~20 ms steady state.
        Reporting that as "the cost of a coarse pass" would be wrong by 18x.
        """
        if (h, w) in self._warm:
            return
        d = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(n):
            a, b = self._to_tensor(d), self._to_tensor(d)
            padder = frame_utils.InputPadder(a.shape, padding_factor=16)
            a, b = padder.pad(a, b)
            self._prep(self.v3, a.shape[-2], a.shape[-1])
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.amp):
                st = self.v3.infer_coarse_state(a, b)
                q = torch.zeros(1, 1024, 2, device=self.dev)
                self.v3.decode_queries(st, query_coords=q)
            if self.v2 is not None:
                self._prep(self.v2, a.shape[-2], a.shape[-1])
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.amp):
                    self.v2(a, b)
        if self.dev.type == 'cuda':
            torch.cuda.synchronize()
        self._warm.add((h, w))

    def _prep(self, model, h, w):
        key = (id(model), h, w)
        if self._shape.get('key') != key:
            model.init_bhwd(1, h, w, self.dev, amp=self.amp)
            self._shape['key'] = key

    def _to_tensor(self, img):
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()[None]
        return t.to(self.dev)

    def coarse(self, img0, img1, key=None, iters_s16=1, iters_s8=8):
        """Full-frame coarse pass, cached per frame pair."""
        if key is not None and key == self.state_key:
            return self.state
        a, b = self._to_tensor(img0), self._to_tensor(img1)
        padder = frame_utils.InputPadder(a.shape, padding_factor=16)
        a, b = padder.pad(a, b)
        self._prep(self.v3, a.shape[-2], a.shape[-1])
        if self.dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.amp):
            st = self.v3.infer_coarse_state(a, b, iters_s16=iters_s16, iters_s8=iters_s8)
        if self.dev.type == 'cuda':
            torch.cuda.synchronize()
        self.coarse_ms = (time.perf_counter() - t0) * 1000
        st['_padder'] = padder
        self.state, self.state_key = st, key
        return st

    def query_region(self, state, x0, y0, w, h, stride=1):
        """Decode only inside the box. Exact; costs decode time only."""
        dev = self.dev
        ys = torch.arange(y0, y0 + h, stride, dtype=torch.float32, device=dev)
        xs = torch.arange(x0, x0 + w, stride, dtype=torch.float32, device=dev)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        q = torch.stack([gx, gy], -1).reshape(1, -1, 2)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.amp):
            if self.uncertainty:
                flow, b = self.v3.decode_queries(state, query_coords=q, return_uncertainty=True)
            else:
                flow, b = self.v3.decode_queries(state, query_coords=q), None
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        f = flow[0].reshape(len(ys), len(xs), 2).float().cpu().numpy()
        bb = b[0].reshape(len(ys), len(xs)).float().cpu().numpy() if b is not None else None
        if stride > 1:
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
            if bb is not None:
                bb = cv2.resize(bb, (w, h), interpolation=cv2.INTER_LINEAR)
        return f, bb, ms

    def crop_region(self, img0, img1, x0, y0, w, h, margin, iters_s16=1, iters_s8=8,
                    stride=2):
        """Run the whole pipeline on the box + margin. Cost scales with area."""
        H, W = img0.shape[:2]
        mx, my = int(w * margin), int(h * margin)
        cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
        cx1, cy1 = min(W, x0 + w + mx), min(H, y0 + h + my)
        c0, c1 = img0[cy0:cy1, cx0:cx1], img1[cy0:cy1, cx0:cx1]

        a, b = self._to_tensor(c0), self._to_tensor(c1)
        padder = frame_utils.InputPadder(a.shape, padding_factor=16)
        a, b = padder.pad(a, b)
        self._prep(self.v3, a.shape[-2], a.shape[-1])
        sync = (lambda: torch.cuda.synchronize()) if self.dev.type == 'cuda' else (lambda: None)
        sync(); t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.amp):
            st = self.v3.infer_coarse_state(a, b, iters_s16=iters_s16, iters_s8=iters_s8)
        sync(); t1 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.amp):
            dense = self.v3.decode_dense_fast(st, stride=stride)
        sync(); t2 = time.perf_counter()
        dense = padder.unpad(dense[0]).float().cpu().numpy().transpose(1, 2, 0)
        f = dense[y0 - cy0:y0 - cy0 + h, x0 - cx0:x0 - cx0 + w]
        # coarse and decode reported separately: the only thing CROP saves over
        # QUERY is the coarse pass, and only in proportion to the cropped area.
        return f, (t1 - t0) * 1000, (t2 - t1) * 1000, (cx1 - cx0) * (cy1 - cy0)

    def full_frame_v2(self, img0, img1):
        if self.v2 is None:
            return None, 0.0
        a, b = self._to_tensor(img0), self._to_tensor(img1)
        padder = frame_utils.InputPadder(a.shape, padding_factor=16)
        a, b = padder.pad(a, b)
        self._prep(self.v2, a.shape[-2], a.shape[-1])
        if self.dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.amp):
            out = self.v2(a, b)[-1]
        if self.dev.type == 'cuda':
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        return padder.unpad(out[0]).float().cpu().numpy().transpose(1, 2, 0), ms
