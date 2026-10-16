#!/usr/bin/env python3
"""Turn a regression log (stdout of tb/run_regression.py) into docs/regression_results.md.

Usage: python3 tb/run_regression.py ... | tee regression.log
       python3 scripts/regression_summary.py regression.log
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
log = Path(sys.argv[1]).read_text()

pat = re.compile(r"\[(\d+x\d+) pipe=(\d)\] jobs passed (\d+)/(\d+), failed (\d+); lockstep cycles (\d+), "
                 r"mismatches (\d+); RTL assertion failures (\d+); coverage ([\d.]+) %")
rows = {}
for m in pat.finditer(log):
    key = (m.group(1), m.group(2))
    rows.setdefault(key, []).append(m.groups())

suite = re.findall(r"=== (\S+): (\d+)/(\d+) tests passed ===", log)

lines = ["| Config | PIPE_MAC | Jobs checked | Cycles compared with C++ model | Model mismatches | RTL assertion failures | Functional coverage | Tests |",
         "|---|---|---:|---:|---:|---:|---:|---|"]
tot_jobs = tot_cycles = 0
for (cfg, pipe), runs in rows.items():
    jobs = sum(int(r[2]) for r in runs)
    cycles = sum(int(r[5]) for r in runs)
    mism = sum(int(r[6]) for r in runs)
    asserts = sum(int(r[7]) for r in runs)
    cov = max(float(r[8]) for r in runs)
    name = cfg + ("p" if pipe == "1" else "")
    tests = next((f"{p}/{t} passed" for n, p, t in suite if n == name), "?")
    tot_jobs += jobs
    tot_cycles += cycles
    lines.append(f"| {cfg} | {pipe} | {jobs} | {cycles} | {mism} | {asserts} | {cov:.0f} % | {tests} |")
lines.append(f"| **Total** | | **{tot_jobs}** | **{tot_cycles}** | | | | |")
text = "\n".join(lines) + "\n"
(ROOT / "docs" / "regression_results.md").write_text(text)
print(text)
