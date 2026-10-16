#!/usr/bin/env python3
"""Build the RTL with Icarus and run the cocotb/pyuvm tests for several array sizes.

Usage:
  python3 tb/run_regression.py                 # default regression
  python3 tb/run_regression.py --sizes 4x4 --jobs 50 --seed 7
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent
RTL_FILES = ("pe.sv", "systolic_array.sv", "sa_top.sv")


def run_size(rows: int, cols: int, pipe: int, jobs: int, seed: int, max_k: int, testcase=None,
             rtl_dir: Path = ROOT / "rtl", tag: str = "") -> bool:
    RTL = [Path(rtl_dir) / f for f in RTL_FILES]
    name = f"{rows}x{cols}" + ("p" if pipe else "")
    build_dir = ROOT / "sim_build" / f"{tag}{name}"
    runner = get_runner("icarus")
    runner.build(
        sources=RTL,
        hdl_toplevel="sa_top",
        parameters={"ROWS": rows, "COLS": cols, "PIPE_MAC": pipe},
        defines={"SA_CHECKS_ON": 1},
        build_args=["-g2012"],
        timescale=("1ns", "1ps"),
        build_dir=build_dir,
        always=True,
    )
    env = {
        "SA_ROWS": str(rows), "SA_COLS": str(cols), "SA_PIPE": str(pipe), "SA_JOBS": str(jobs),
        "SA_SEED": str(seed), "SA_MAX_K": str(max_k),
    }
    os.environ.update(env)
    results = runner.test(
        hdl_toplevel="sa_top",
        test_module="test_sa_top",
        test_dir=ROOT / "tb",
        build_dir=build_dir,
        extra_env=env,
        testcase=testcase,
        results_xml=f"results_{name}.xml",
    )
    # get_results returns (num_tests, num_failed)
    from cocotb_tools.runner import get_results
    total, failed = get_results(results)
    print(f"=== {name}: {total - failed}/{total} tests passed ===", flush=True)
    return failed == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4x4,8x8,3x5,1x1,4x4p,8x8p,1x1p",
                    help="comma list of ROWSxCOLS, suffix p for PIPE_MAC=1")
    ap.add_argument("--jobs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-k", type=int, default=256)
    ap.add_argument("--testcase", default=None)
    ap.add_argument("--rtl-dir", default=str(ROOT / "rtl"))
    ap.add_argument("--tag", default="", help="prefix for the build directory")
    args = ap.parse_args()

    subprocess.run(["make", "-C", str(ROOT / "model"), "libsamodel.so"], check=True)

    ok = True
    for s in args.sizes.split(","):
        s = s.strip().lower()
        pipe = 1 if s.endswith("p") else 0
        r, c = (int(x) for x in s.rstrip("p").split("x"))
        ok &= run_size(r, c, pipe, args.jobs, args.seed, args.max_k, args.testcase,
                       Path(args.rtl_dir), args.tag)
    print("REGRESSION", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
