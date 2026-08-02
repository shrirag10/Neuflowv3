"""Stream-extract FlyingChairs.zip, converting .ppm -> .png on the fly.

Reads the zip from a slow source (rclone mount) sequentially, writes
.flo files unchanged and images as lossless PNG, never storing any .ppm
on disk. Peak local usage ~46 GB instead of ~63 GB.

Usage:
    python3 scripts/extract_chairs_png.py \
        --zip ~/OneDrive/FlyingChairs/FlyingChairs.zip \
        --out datasets/
"""

import argparse
import io
import os
import zipfile

import cv2
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--out', default='datasets/')
    args = parser.parse_args()

    zip_path = os.path.expanduser(args.zip)

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for info in tqdm(members, unit='file'):
            if info.is_dir():
                continue
            dest = os.path.join(args.out, info.filename)
            os.makedirs(os.path.dirname(dest), exist_ok=True)

            if info.filename.endswith('.ppm'):
                dest = dest[:-4] + '.png'
                if os.path.exists(dest):
                    continue
                data = zf.read(info)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite(dest, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
                    continue
                with zf.open(info) as src, open(dest, 'wb') as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)


if __name__ == '__main__':
    main()
