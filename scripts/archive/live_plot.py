import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
live_plot.py — Real-time training dashboard for NeuFlow v3
Reads checkpoints/neuflowv3/train_log.csv and serves a live plot at localhost:5000
Open http://localhost:5000 in your browser. Refreshes every 5 seconds.
"""

import csv, os, time
from http.server import HTTPServer, BaseHTTPRequestHandler

LOG_PATH = 'checkpoints/neuflowv3/train_log.csv'
REFRESH_S = 5
PORT = 5000


def read_log():
    if not os.path.exists(LOG_PATH):
        return [], [], [], []
    steps, epe, loss, mag = [], [], [], []
    try:
        with open(LOG_PATH, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row['step']))
                epe.append(float(row['epe']))
                loss.append(float(row['loss']))
                mag.append(float(row['mag']))
    except Exception:
        pass
    return steps, epe, loss, mag


def smooth(values, w=50):
    out = []
    for i, v in enumerate(values):
        s = max(0, i - w + 1)
        out.append(sum(values[s:i+1]) / (i - s + 1))
    return out


def build_html(steps, epe, loss, mag):
    total = 30000
    pct = f"{steps[-1]/total*100:.1f}" if steps else "0"
    cur_epe = f"{epe[-1]:.4f}" if epe else "—"
    cur_loss = f"{loss[-1]:.4f}" if loss else "—"
    eta = "—"
    if len(steps) >= 2:
        rate = (steps[-1] - steps[0]) / max(len(steps) - 1, 1)  # steps per row
        remaining = total - steps[-1]
        # assume ~6 steps/sec
        secs = remaining / 6
        h, m = int(secs // 3600), int((secs % 3600) // 60)
        eta = f"{h}h {m}m"

    # Build sparkline data strings
    sm_epe = smooth(epe, 100)
    sm_loss = smooth(loss, 100)

    def pts(xs, ys, h=120, pad=10):
        if not xs: return ""
        mn, mx = min(ys), max(ys)
        rng = mx - mn or 1
        pts_list = []
        W = 700
        x0, x1 = xs[0], xs[-1]
        xrng = x1 - x0 or 1
        for x, y in zip(xs, ys):
            px = pad + (x - x0) / xrng * (W - 2*pad)
            py = pad + (1 - (y - mn) / rng) * (h - 2*pad)
            pts_list.append(f"{px:.1f},{py:.1f}")
        return " ".join(pts_list)

    epe_pts  = pts(steps, sm_epe)
    loss_pts = pts(steps, sm_loss)

    epe_raw  = pts(steps, epe)
    loss_raw = pts(steps, loss)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{REFRESH_S}">
<title>NeuFlow v3 Training</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f0f1a; color: #e0e0ff; font-family: 'Segoe UI', system-ui, sans-serif; padding: 24px; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; color: #a78bfa; margin-bottom: 6px; }}
  .sub {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }}
  .card {{ background: #1a1a2e; border-radius: 12px; padding: 16px 20px; border: 1px solid #2d2d4e; }}
  .card .label {{ font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; }}
  .card .val {{ font-size: 1.8rem; font-weight: 700; margin-top: 4px; }}
  .card .val.purple {{ color: #a78bfa; }}
  .card .val.green  {{ color: #34d399; }}
  .card .val.orange {{ color: #fb923c; }}
  .card .val.blue   {{ color: #60a5fa; }}
  .chart-box {{ background: #1a1a2e; border-radius: 12px; padding: 20px; border: 1px solid #2d2d4e; margin-bottom: 16px; }}
  .chart-box h2 {{ font-size: 0.9rem; color: #9ca3af; margin-bottom: 12px; text-transform: uppercase; letter-spacing: .05em; }}
  svg {{ width: 100%; height: auto; display: block; }}
  .prog-bar {{ background: #2d2d4e; border-radius: 8px; height: 8px; overflow: hidden; margin-bottom: 8px; }}
  .prog-fill {{ background: linear-gradient(90deg, #7c3aed, #a78bfa); height: 100%; border-radius: 8px;
                width: {pct}%; transition: width .5s ease; }}
  .prog-label {{ font-size: 0.8rem; color: #6b7280; }}
  .refresh-note {{ font-size: 0.75rem; color: #374151; margin-top: 24px; text-align: right; }}
</style>
</head>
<body>
<h1>⚡ NeuFlow v3 — Live Training</h1>
<p class="sub">InfiniDepth-aligned · End-to-end · lr=1e-5 · 3-scale fusion · Auto-refreshes every {REFRESH_S}s</p>

<div class="cards">
  <div class="card"><div class="label">Current EPE</div><div class="val green">{cur_epe}</div></div>
  <div class="card"><div class="label">Current Loss</div><div class="val orange">{cur_loss}</div></div>
  <div class="card"><div class="label">Progress</div><div class="val purple">{pct}%</div></div>
  <div class="card"><div class="label">ETA</div><div class="val blue">{eta}</div></div>
</div>

<div class="chart-box">
  <h2>Progress — {steps[-1] if steps else 0} / {total} steps</h2>
  <div class="prog-bar"><div class="prog-fill"></div></div>
  <div class="prog-label">{steps[-1] if steps else 0} / {total} steps</div>
</div>

<div class="chart-box">
  <h2>EPE (End-Point Error) ↓ lower is better</h2>
  <svg viewBox="0 0 720 140" preserveAspectRatio="none">
    <rect width="720" height="140" fill="#13131f" rx="6"/>
    {"<polyline points='" + epe_raw + "' fill='none' stroke='#34d39930' stroke-width='1'/>" if epe_raw else ""}
    {"<polyline points='" + epe_pts + "' fill='none' stroke='#34d399' stroke-width='2' stroke-linejoin='round'/>" if epe_pts else ""}
    <text x="10" y="20" fill="#4b5563" font-size="11">EPE</text>
    {"<text x='10' y='135' fill='#4b5563' font-size='10'>" + str(steps[0]) + "</text><text x='660' y='135' fill='#4b5563' font-size='10'>" + str(steps[-1]) + "</text>" if steps else ""}
  </svg>
</div>

<div class="chart-box">
  <h2>Loss ↓ lower is better</h2>
  <svg viewBox="0 0 720 140" preserveAspectRatio="none">
    <rect width="720" height="140" fill="#13131f" rx="6"/>
    {"<polyline points='" + loss_raw + "' fill='none' stroke='#fb923c30' stroke-width='1'/>" if loss_raw else ""}
    {"<polyline points='" + loss_pts + "' fill='none' stroke='#fb923c' stroke-width='2' stroke-linejoin='round'/>" if loss_pts else ""}
    <text x="10" y="20" fill="#4b5563" font-size="11">Loss</text>
  </svg>
</div>

<p class="refresh-note">Last updated: {time.strftime('%H:%M:%S')} · Refreshes every {REFRESH_S}s</p>
</body>
</html>"""
    return html


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        steps, epe, loss, mag = read_log()
        page = build_html(steps, epe, loss, mag)
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(page.encode())

    def log_message(self, *_):  # suppress request logs
        pass


if __name__ == '__main__':
    print(f"Dashboard → http://localhost:{PORT}")
    print(f"Reading: {LOG_PATH}")
    print("Ctrl+C to stop")
    HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
