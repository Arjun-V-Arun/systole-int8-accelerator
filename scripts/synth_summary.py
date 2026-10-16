#!/usr/bin/env python3
"""Collect synth/out/*/ results into docs/synth_results.md."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "synth" / "out"

rows = []
for d in sorted(OUT.iterdir()):
    stat = (d / "stat.txt").read_text()
    log = (d / "yosys.log").read_text()
    area = float(re.search(r"Chip area for module .*?: ([\d.]+)", stat).group(1))
    cells = int(re.search(r"Number of cells:\s+(\d+)", stat).group(1))
    flops = sum(int(n) for n in re.findall(r"sky130_fd_sc_hd__df\w+\s+(\d+)", stat))
    delay = float(re.findall(r"Delay =\s*([\d.]+) ps", log)[-1])
    start = re.findall(r"Start-point = \S+ \((.*?)\)\.", log)[-1]
    m = re.match(r"(\d+)x(\d+)(p?)_t(\d+)", d.name)
    r, c, p, t = int(m.group(1)), int(m.group(2)), bool(m.group(3)), int(m.group(4))
    start = start.lstrip("\\")
    start = re.sub(r"\\?u_array\.g_row\[\d+\]\.g_col\[\d+\]\.u_pe\.", "PE.", start)
    rows.append((r, c, p, t, cells, flops, area, delay, start))

lines = [
    "| Array | PIPE_MAC | ABC mapping | Cells | Flip-flops | Area (mm²) | Logic delay (ns) | Est. fmax (MHz) | Critical path starts at |",
    "|---|---|---|---:|---:|---:|---:|---:|---|",
]
for r, c, p, t, cells, flops, area, delay, start in sorted(rows, key=lambda x: (x[0], x[1], x[3] != 0, x[2])):
    fmax = 1e3 / (delay / 1e3 + 0.5)  # +0.5 ns rough allowance for clock-to-Q and setup
    tgt = "area" if t == 0 else f"delay, {t / 1000:.1f} ns target"
    lines.append(f"| {r}x{c} | {int(p)} | {tgt} | {cells} | {flops} | {area / 1e6:.3f} | "
                 f"{delay / 1000:.2f} | {fmax:.0f} | `{start}` |")
text = "\n".join(lines) + "\n"
(ROOT / "docs" / "synth_results.md").write_text(text)
print(text)
