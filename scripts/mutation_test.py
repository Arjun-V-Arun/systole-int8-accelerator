#!/usr/bin/env python3
"""Mutation testing: inject realistic bugs into a copy of the RTL and check
that the verification environment catches every one of them.

A testbench that passes on correct RTL proves little on its own. This script
shows the environment fails when the design is wrong, and reports which
checker caught each bug:
  scoreboard  wrong numerical result
  lockstep    cycle-level disagreement with the C++ model
  assertion   an RTL assertion (SA_CHECKS_ON) fired
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTANTS = [
    ("sign_extension_dropped",
     "pe.sv", "{{(ACC_W-2*DW){addend[2*DW-1]}}, addend}", "{{(ACC_W-2*DW){1'b0}}, addend}",
     "negative products are zero-extended into the accumulator"),
    ("unsigned_multiply",
     "pe.sv", "assign prod = a_in * b_in;", "assign prod = $unsigned(a_in) * $unsigned(b_in);",
     "operands multiplied as unsigned"),
    ("pe_ignores_stall",
     "pe.sv", "end else if (step) begin", "end else begin",
     "PEs keep advancing while the array should be stalled"),
    ("flush_one_short",
     "sa_top.sv", "localparam int FLUSH_STEPS = ROWS + COLS - 2 + PIPE_MAC;",
     "localparam int FLUSH_STEPS = ROWS + COLS - 3 + PIPE_MAC;",
     "last operand never reaches the bottom-right PE"),
    ("skew_wrong_tap",
     "sa_top.sv", "assign a_edge[r*DW +: DW] = sr[r-1];", "assign a_edge[r*DW +: DW] = sr[0];",
     "rows below row 1 get the wrong delay"),
    ("no_clear_between_jobs",
     "sa_top.sv", "assign clear     = out_fire && out_last;", "assign clear     = 1'b0;",
     "accumulators carry over into the next job"),
    ("drain_ignores_ready",
     "sa_top.sv", "                    if (out_fire) begin", "                    if (out_valid) begin",
     "output rows advance without a handshake"),
    ("ready_during_flush",
     "sa_top.sv", "assign in_ready  = (state == S_LOAD);", "assign in_ready  = (state != S_DRAIN);",
     "input accepted while the array is flushing"),
    ("pipe_missing_flush_step",
     "sa_top.sv", "localparam int FLUSH_STEPS = ROWS + COLS - 2 + PIPE_MAC;",
     "localparam int FLUSH_STEPS = ROWS + COLS - 2;",
     "pipelined MAC: last product left in the product register"),
    ("pipe_reg_not_cleared",
     "pe.sv", "else if (clear)  prod_q <= '0;", "else if (1'b0)   prod_q <= '0;",
     "pipelined MAC: product register not cleared between jobs"),
]

# Equivalent mutants: the change cannot alter any output, so no test can
# catch it. Each one is listed with the reason, which was worked out after
# the mutant first came back as MISSED.
EQUIVALENT = {
    "pipe_reg_not_cleared":
        "On the final flush step every PE multiplies zeros (the last real operands "
        "have already moved past it), so prod_q is already 0 when clear fires. "
        "The clear is redundant in normal operation and kept only as a safety net.",
}

# Mutants that only change the pipelined datapath are run with PIPE_MAC=1.
SIZE_FOR = {"pipe_missing_flush_step": "4x4p", "pipe_reg_not_cleared": "4x4p"}


def classify(log: str) -> list[str]:
    caught = []
    if "MISMATCH" in log or "wrong result after reset" in log:
        caught.append("scoreboard")
    if "disagrees with RTL" in log or "!= model" in log:
        caught.append("lockstep")
    if "[sa_top]" in log or "RTL assertions fired" in log:
        caught.append("assertion")
    return caught


def main():
    out_dir = ROOT / "sim_build" / "mutants"
    shutil.rmtree(out_dir, ignore_errors=True)
    rows = []
    all_caught = True
    for name, fname, old, new, desc in MUTANTS:
        mdir = out_dir / name
        shutil.copytree(ROOT / "rtl", mdir)
        src = (mdir / fname).read_text()
        assert src.count(old) >= 1, f"mutation anchor not found for {name}"
        (mdir / fname).write_text(src.replace(old, new, 1))
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tb" / "run_regression.py"), "--sizes", SIZE_FOR.get(name, "4x4"),
             "--jobs", "60", "--rtl-dir", str(mdir), "--tag", f"mut_{name}_"],
            capture_output=True, text=True, timeout=1800)
        log = proc.stdout + proc.stderr
        (out_dir / f"{name}.log").write_text(log)
        failed = proc.returncode != 0
        caught_by = classify(log) if failed else []
        if failed:
            result = "CAUGHT"
        elif name in EQUIVALENT:
            result = "EQUIVALENT"
            caught_by = ["n/a: " + EQUIVALENT[name]]
        else:
            result = "MISSED"
            all_caught = False
        rows.append((name, desc, result, ", ".join(caught_by) or "-"))
        print(f"{name:24s} {result:10s} {', '.join(caught_by)[:60] or '-'}", flush=True)

    md = ["| Injected bug | Effect | Result | Caught by |", "|---|---|---|---|"]
    md += [f"| `{n}` | {d} | {r} | {c} |" for n, d, r, c in rows]
    (ROOT / "docs" / "mutation_results.md").write_text("\n".join(md) + "\n")
    n_eq = sum(r[2] == "EQUIVALENT" for r in rows)
    print(f"\n{sum(r[2] == 'CAUGHT' for r in rows)}/{len(rows) - n_eq} non-equivalent mutants caught, "
          f"{n_eq} equivalent")
    sys.exit(0 if all_caught else 1)


if __name__ == "__main__":
    main()
