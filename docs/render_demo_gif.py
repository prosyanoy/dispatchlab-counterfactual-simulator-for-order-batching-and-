"""Record `make demo` and render docs/demo.gif (real output, time-compressed to ~60 s).

Usage (from the repo root, venv active): python docs/render_demo_gif.py
The stage lines and report excerpt in the GIF are the actual output of the run; only the
timeline is compressed so the whole demo plays in about a minute.
"""
from __future__ import annotations

import subprocess
import textwrap
import time
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
TOTAL_S, TYPE_S, END_S = 60.0, 2.0, 9.0
COLS, ROWS, FS = 112, 27, 15
BG, FG, DIM, GREEN, BLUE = "#1b1b1a", "#e8e7e3", "#8d8c85", "#5fd38d", "#6ab0ff"


def record() -> list[tuple[float, str]]:
    t0, events = time.time(), []
    p = subprocess.Popen(["make", "demo"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        events.append((time.time() - t0, line.rstrip("\n")))
    if p.wait() != 0:
        raise SystemExit("make demo failed")
    events.append((time.time() - t0, ""))
    return events


def report_excerpt() -> list[str]:
    text = (ROOT / "outputs" / "report.md").read_text().splitlines()
    out = []
    for line in text:
        if line.startswith("## 2."):
            break
        if line.startswith("!") or not line.strip() or line.startswith("Decision rule") or line.startswith("21 simulated"):
            continue
        out.extend(textwrap.wrap(line.replace("**", "").replace("`", ""), COLS - 2) if not line.startswith("|") else [line])
    return out[:14]


def main() -> None:
    events = record()
    run_s = events[-1][0]
    scale = (TOTAL_S - TYPE_S - END_S) / run_s
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf"), FS)
    cw, ch = int(font.getlength("M")), FS + 6
    W, H = cw * COLS + 24, ch * (ROWS + 2) + 16
    cmd = "make demo"
    excerpt = report_excerpt()

    def frame(lines: list[tuple[str, str]], status: str) -> Image.Image:
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        for i, (txt, col) in enumerate(lines[-ROWS:]):
            d.text((12, 8 + i * ch), txt[:COLS], font=font, fill=col)
        d.text((12, H - ch - 4), status, font=font, fill=DIM)
        return im

    frames, durs = [], []

    def add(im, ms):
        frames.append(im)
        durs.append(ms)

    prompt = "$ "
    for k in range(len(cmd) + 1):  # typing the command
        add(frame([(prompt + cmd[:k] + "_", FG)], "recording: make demo (real output)"), int(TYPE_S * 1000 / (len(cmd) + 1)))
    step = 1.0  # demo seconds per frame
    n = int((TOTAL_S - TYPE_S - END_S) / step)
    for i in range(1, n + 1):
        real_t = i * step / scale
        shown = [(prompt + cmd, FG)] + [
            (t, GREEN if t.startswith("[") else FG) for ts, t in events[1:] if ts <= real_t and t
        ]
        add(frame(shown, f"elapsed {real_t:5.0f}s real  |  playback {i * step:4.0f}s  |  time-compressed {1 / scale:.1f}x"), int(step * 1000))
    done = [(prompt + cmd, FG)] + [(t, GREEN if t.startswith("[") else FG) for _, t in events[1:] if t]
    done += [("", FG), (f"$ head -n {len(excerpt) + 3} outputs/report.md   # excerpt", FG)] + [
        (l, BLUE if l.startswith("#") else FG) for l in excerpt
    ]
    add(frame(done, f"finished in {run_s:.0f}s real time (shown time-compressed)"), int(END_S * 1000))
    out = ROOT / "docs" / "demo.gif"
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=32) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB, {sum(durs) / 1000:.0f}s, real run {run_s:.0f}s)")


if __name__ == "__main__":
    main()
