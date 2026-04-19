#!/bin/bash
# Download VKITTI2 RGB + Forward Flow for optical flow training.
#
# Data source: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
# License: CC BY-NC-SA 3.0
#
# RGB:         ~7 GB  (extracted: ~7 GB)
# ForwardFlow: ~30 GB (extracted: ~5 GB)
#
# Strategy: Stream-extract directly from HTTP → tar, so the .tar
#           is NEVER written to disk.  This keeps disk usage to just
#           the extracted files (~12 GB total).

set -e

DEST_DIR="datasets/vkitti2"
mkdir -p "$DEST_DIR"

RGB_URL="https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar"
FLOW_URL="https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_forwardFlow.tar"

# ---------------------------------------------------------------------------
#  stream_extract URL LABEL DEST
#    Downloads URL while showing a tqdm progress bar (LABEL),
#    and pipes the bytes directly into  tar -xf - -C DEST
# ---------------------------------------------------------------------------
stream_extract() {
    local url="$1" label="$2" dest="$3"
    python3 -c "
import sys, urllib.request, subprocess

from tqdm import tqdm

url   = sys.argv[1]
label = sys.argv[2]
dest  = sys.argv[3]

# Open HTTP stream
resp = urllib.request.urlopen(url)
total = int(resp.headers.get('Content-Length', 0))

# Spawn tar reading from stdin
tar = subprocess.Popen(['tar', '-xf', '-', '-C', dest], stdin=subprocess.PIPE)

CHUNK = 1 << 20   # 1 MiB
with tqdm(total=total, unit='B', unit_scale=True, desc=label) as bar:
    while True:
        chunk = resp.read(CHUNK)
        if not chunk:
            break
        tar.stdin.write(chunk)
        bar.update(len(chunk))

tar.stdin.close()
rc = tar.wait()
if rc != 0:
    sys.exit(rc)
" "$url" "$label" "$dest"
}

echo "=== Downloading & extracting VKITTI2 RGB (~7 GB) ==="
stream_extract "$RGB_URL" "RGB" "$DEST_DIR"

echo ""
echo "=== Downloading & extracting VKITTI2 Forward Flow (~30 GB) ==="
stream_extract "$FLOW_URL" "Flow" "$DEST_DIR"

echo ""
echo "=== Done! ==="
echo "Dataset extracted to: $DEST_DIR"
du -sh "$DEST_DIR"
