"""Stream-extract FlyingChairs from the Freiburg server in ONE HTTP stream —
the 31 GB zip is never stored on disk.

.flo files are written as-is; .ppm images are converted to lossless PNG on
the fly. Peak disk usage ~50 GB instead of ~94 GB (zip + raw extraction).
On connection failure the stream restarts but already-written files are
skipped (bytes are re-downloaded, disk work is not repeated).

Usage:
    python3 scripts/stream_chairs_png.py --out datasets/
"""

import argparse
import os
import shutil
import time

import cv2
import numpy as np
import requests
from stream_unzip import stream_unzip
from tqdm import tqdm

URL = 'https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip'
TOTAL_FILES = 68617   # from archive listing (for tqdm)
MIN_FREE_GB = 5


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def http_chunks():
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        yield from r.iter_content(chunk_size=1 << 20)


def drain(chunks):
    for _ in chunks:
        pass


def extract_all(out_dir):
    n_new = 0
    pbar = tqdm(total=TOTAL_FILES, unit='file')
    for name_b, size, chunks in stream_unzip(http_chunks()):
        pbar.update(1)
        name = name_b.decode()
        if name.endswith('/'):
            drain(chunks)
            continue
        if free_gb(out_dir) < MIN_FREE_GB:
            raise RuntimeError(f'Aborting: <{MIN_FREE_GB} GB free disk space')

        dest = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        if name.endswith('.ppm'):
            dest = dest[:-4] + '.png'
            if os.path.exists(dest):
                drain(chunks)
                continue
            data = b''.join(chunks)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            tmp = dest + '.tmp'
            cv2.imwrite(tmp + '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 5])
            os.replace(tmp + '.png', dest)
            n_new += 1
        else:
            if size is not None and os.path.exists(dest) and os.path.getsize(dest) == size:
                drain(chunks)
                continue
            tmp = dest + '.tmp'
            with open(tmp, 'wb') as f:
                for c in chunks:
                    f.write(c)
            os.replace(tmp, dest)
            n_new += 1
    pbar.close()
    return n_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='datasets/')
    parser.add_argument('--retries', type=int, default=10)
    args = parser.parse_args()

    for attempt in range(args.retries):
        try:
            n = extract_all(args.out)
            print(f'DONE: wrote {n} new files')
            return
        except (RuntimeError, KeyboardInterrupt):
            raise
        except Exception as e:
            wait = min(120, 10 * (attempt + 1))
            print(f'[retry {attempt + 1}/{args.retries}] {type(e).__name__}: {e} — restarting stream in {wait}s')
            time.sleep(wait)

    raise SystemExit('FAILED: retries exhausted')


if __name__ == '__main__':
    main()
