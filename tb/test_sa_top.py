"""UVM-style verification environment for rtl/sa_top.sv (pyuvm + cocotb).

Structure
---------
  SaTest
   └── SaEnv
        ├── InAgent:  sequencer -> InDriver (random input bubbles, garbage data when invalid)
        │             InMonitor (rebuilds A and B from accepted beats)
        ├── OutAgent: OutReadyDriver (random backpressure)
        │             OutMonitor (rebuilds C from accepted rows)
        ├── Scoreboard      NumPy reference vs. DUT result, job by job, in order
        ├── LockstepChecker C++ cycle-accurate model vs. DUT, every output on every cycle
        └── Coverage        functional coverage on job shape, data corners and stall scenarios

Two independent checks run at the same time. The scoreboard says the answer
is right. The lockstep checker says the timing is right, cycle for cycle,
including every stall. On top of that the RTL has its own assertions
(SA_CHECKS_ON) and the test fails if any of them fired.

Knobs (environment variables, set by tb/run_regression.py):
  SA_ROWS, SA_COLS, SA_PIPE, SA_JOBS, SA_SEED, SA_GAP_PROB, SA_BP_PROB, SA_MAX_K
"""

import os
import random

import cocotb
import numpy as np
import pyuvm
from cocotb.clock import Clock
from cocotb.simtime import get_sim_time
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from pyuvm import (
    ConfigDB,
    uvm_analysis_port,
    uvm_component,
    uvm_driver,
    uvm_env,
    uvm_sequence,
    uvm_sequence_item,
    uvm_sequencer,
    uvm_subscriber,
    uvm_test,
    uvm_tlm_analysis_fifo,
)

from model_bridge import CycleModel
from sa_utils import pack_int8, reference_matmul, unpack_int8, unpack_int32

ROWS = int(os.environ.get("SA_ROWS", "4"))
COLS = int(os.environ.get("SA_COLS", "4"))
N_JOBS = int(os.environ.get("SA_JOBS", "200"))
SEED = int(os.environ.get("SA_SEED", "1"))
GAP_PROB = float(os.environ.get("SA_GAP_PROB", "0.25"))
BP_PROB = float(os.environ.get("SA_BP_PROB", "0.3"))
MAX_K = int(os.environ.get("SA_MAX_K", "256"))
PIPE = int(os.environ.get("SA_PIPE", "0"))
CLK_NS = 10


def cycle_now() -> int:
    return int(get_sim_time(unit="ns")) // CLK_NS


def bit(sig) -> int:
    return int(sig.value)


# ---------------------------------------------------------------------------
# Transactions and sequences
# ---------------------------------------------------------------------------
class MatmulItem(uvm_sequence_item):
    """One job: A is ROWS x K, B is K x COLS, both signed INT8."""

    def __init__(self, name="matmul", a=None, b=None, kind="random"):
        super().__init__(name)
        self.a = a
        self.b = b
        self.kind = kind

    @property
    def k(self) -> int:
        return self.a.shape[1]

    def __str__(self):
        return f"MatmulItem(kind={self.kind}, K={self.k})"


def make_job(kind: str, k: int, rng: np.random.Generator) -> MatmulItem:
    shape_a, shape_b = (ROWS, k), (k, COLS)
    if kind == "random":
        a = rng.integers(-128, 128, shape_a)
        b = rng.integers(-128, 128, shape_b)
    elif kind == "all_min":          # -128 * -128: largest positive product
        a = np.full(shape_a, -128)
        b = np.full(shape_b, -128)
    elif kind == "all_max":          # 127 * 127
        a = np.full(shape_a, 127)
        b = np.full(shape_b, 127)
    elif kind == "min_times_max":    # -128 * 127: most negative product
        a = np.full(shape_a, -128)
        b = np.full(shape_b, 127)
    elif kind == "extremes":         # random mix of corner values
        corners = np.array([-128, -127, -1, 0, 1, 126, 127])
        a = rng.choice(corners, shape_a)
        b = rng.choice(corners, shape_b)
    elif kind == "zeros":
        a = np.zeros(shape_a, dtype=int)
        b = np.zeros(shape_b, dtype=int)
    elif kind == "sparse":           # 90 % zeros
        a = rng.integers(-128, 128, shape_a) * (rng.random(shape_a) < 0.1)
        b = rng.integers(-128, 128, shape_b) * (rng.random(shape_b) < 0.1)
    elif kind == "identity":         # A x I = A when K == COLS
        a = rng.integers(-128, 128, shape_a)
        b = np.eye(k, COLS, dtype=int)
    else:
        raise ValueError(kind)
    return MatmulItem(kind, a.astype(np.int64), b.astype(np.int64), kind)


def random_k(rng: np.random.Generator) -> int:
    big = max(ROWS, COLS)
    bucket = rng.choice(4, p=[0.1, 0.2, 0.4, 0.3])
    if bucket == 0:
        return 1
    if bucket == 1:
        return int(rng.integers(2, max(3, ROWS + 1)))
    if bucket == 2:
        return int(rng.integers(ROWS, 4 * big + 1))
    return int(rng.integers(4 * big + 1, MAX_K + 1))


class DirectedSeq(uvm_sequence):
    """Hand-picked corner cases, run without stalls so latency can be checked."""

    async def body(self):
        rng = np.random.default_rng(SEED)
        jobs = [
            ("random", 1), ("all_min", 1), ("random", ROWS), ("random", COLS),
            ("identity", COLS), ("zeros", 5), ("all_max", 17), ("min_times_max", 17),
            ("all_min", MAX_K), ("random", MAX_K), ("extremes", 33), ("sparse", 40),
        ]
        for kind, k in jobs:
            item = make_job(kind, k, rng)
            await self.start_item(item)
            await self.finish_item(item)


class RandomSeq(uvm_sequence):
    """Constrained-random jobs. Shape and data kind are both randomized."""

    async def body(self):
        rng = np.random.default_rng(SEED)
        kinds = ["random", "random", "random", "extremes", "sparse",
                 "all_min", "all_max", "min_times_max", "zeros", "identity"]
        for _ in range(N_JOBS):
            kind = str(rng.choice(kinds))
            k = COLS if kind == "identity" else random_k(rng)
            item = make_job(kind, k, rng)
            await self.start_item(item)
            await self.finish_item(item)


# ---------------------------------------------------------------------------
# Input agent
# ---------------------------------------------------------------------------
class InDriver(uvm_driver):
    def build_phase(self):
        self.dut = cocotb.top
        self.gap_prob = ConfigDB().get(self, "", "GAP_PROB")
        self.rnd = random.Random(SEED + 101)

    async def run_phase(self):
        dut = self.dut
        while True:
            item = await self.seq_item_port.get_next_item()
            for k in range(item.k):
                # Random bubbles. Garbage on the data lines while in_valid is
                # low proves the design ignores them.
                while self.rnd.random() < self.gap_prob:
                    dut.in_valid.value = 0
                    dut.in_a.value = self.rnd.getrandbits(8 * ROWS)
                    dut.in_b.value = self.rnd.getrandbits(8 * COLS)
                    dut.in_last.value = self.rnd.getrandbits(1)
                    await RisingEdge(dut.clk)
                dut.in_valid.value = 1
                dut.in_a.value = pack_int8(item.a[:, k])
                dut.in_b.value = pack_int8(item.b[k, :])
                dut.in_last.value = int(k == item.k - 1)
                while True:
                    await RisingEdge(dut.clk)
                    if bit(dut.in_ready):  # sampled before the edge
                        break
            dut.in_valid.value = 0
            dut.in_last.value = 0
            self.seq_item_port.item_done()


class InMonitor(uvm_component):
    """Rebuilds each job from the beats the DUT actually accepted."""

    def build_phase(self):
        self.ap = uvm_analysis_port("ap", self)
        self.dut = cocotb.top

    async def run_phase(self):
        dut = self.dut
        cols_a, rows_b = [], []
        gap_seen = False
        first_cycle = None
        last_clear_cycle = -10
        while True:
            await RisingEdge(dut.clk)
            if not bit(dut.rst_n):
                cols_a, rows_b, gap_seen, first_cycle = [], [], False, None
                continue
            now = cycle_now()
            ready, valid = bit(dut.in_ready), bit(dut.in_valid)
            if ready and valid:
                if not cols_a:
                    first_cycle = now
                cols_a.append(unpack_int8(int(dut.in_a.value), ROWS))
                rows_b.append(unpack_int8(int(dut.in_b.value), COLS))
                if bit(dut.in_last):
                    a = np.array(cols_a, dtype=np.int64).T
                    b = np.array(rows_b, dtype=np.int64)
                    self.ap.write({
                        "a": a, "b": b, "first_cycle": first_cycle,
                        "in_gap": gap_seen,
                        "back_to_back": first_cycle == last_clear_cycle + 1,
                    })
                    cols_a, rows_b, gap_seen = [], [], False
            elif ready and not valid and cols_a:
                gap_seen = True
            if bit(dut.out_valid) and bit(dut.out_ready) and bit(dut.out_last):
                last_clear_cycle = now


class InAgent(uvm_component):
    def build_phase(self):
        self.seqr = uvm_sequencer("seqr", self)
        self.driver = InDriver("driver", self)
        self.monitor = InMonitor("monitor", self)

    def connect_phase(self):
        self.driver.seq_item_port.connect(self.seqr.seq_item_export)


# ---------------------------------------------------------------------------
# Output agent
# ---------------------------------------------------------------------------
class OutReadyDriver(uvm_component):
    def build_phase(self):
        self.dut = cocotb.top
        self.bp_prob = ConfigDB().get(self, "", "BP_PROB")
        self.rnd = random.Random(SEED + 202)

    async def run_phase(self):
        while True:
            self.dut.out_ready.value = int(self.rnd.random() >= self.bp_prob)
            await RisingEdge(self.dut.clk)


class OutMonitor(uvm_component):
    def build_phase(self):
        self.ap = uvm_analysis_port("ap", self)
        self.dut = cocotb.top
        self.jobs_seen = 0

    async def run_phase(self):
        dut = self.dut
        rows, bp_seen = [], False
        while True:
            await RisingEdge(dut.clk)
            if not bit(dut.rst_n):
                rows, bp_seen = [], False
                continue
            if bit(dut.out_valid):
                if not bit(dut.out_ready):
                    bp_seen = True
                    continue
                rows.append(unpack_int32(int(dut.out_row.value), COLS))
                if bit(dut.out_last):
                    if len(rows) != ROWS:
                        self.logger.error(f"job returned {len(rows)} rows, expected {ROWS}")
                    self.jobs_seen += 1
                    self.ap.write({"c": np.array(rows, dtype=np.int64),
                                   "last_cycle": cycle_now(), "backpressure": bp_seen})
                    rows, bp_seen = [], False


class OutAgent(uvm_component):
    def build_phase(self):
        self.ready_driver = OutReadyDriver("ready_driver", self)
        self.monitor = OutMonitor("monitor", self)


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------
class Scoreboard(uvm_component):
    def build_phase(self):
        self.in_fifo = uvm_tlm_analysis_fifo("in_fifo", self)
        self.out_fifo = uvm_tlm_analysis_fifo("out_fifo", self)
        self.in_export = self.in_fifo.analysis_export
        self.out_export = self.out_fifo.analysis_export
        self.passed = 0
        self.failed = 0
        self.latencies = []  # (K, cycles) for every job

    async def run_phase(self):
        while True:
            job = await self.in_fifo.get()
            res = await self.out_fifo.get()
            expected = reference_matmul(job["a"], job["b"])
            k = job["a"].shape[1]
            self.latencies.append((k, res["last_cycle"] - job["first_cycle"] + 1,
                                   job["in_gap"] or res["backpressure"]))
            if np.array_equal(expected, res["c"]):
                self.passed += 1
            else:
                self.failed += 1
                bad = np.argwhere(expected != res["c"])
                r, c = bad[0]
                self.logger.error(
                    f"MISMATCH K={k}: C[{r}][{c}] expected {expected[r, c]} got {res['c'][r, c]} "
                    f"({len(bad)} wrong elements)")

    def check_phase(self):
        if self.in_fifo.can_get() or self.out_fifo.can_get():
            self.logger.error("scoreboard finished with unmatched transactions")
            self.failed += 1


class LockstepChecker(uvm_component):
    """Runs the C++ cycle model next to the DUT and compares every cycle.

    At each rising edge, cocotb reads values from before the edge. Those are
    exactly the inputs the DUT samples and the outputs it was driving, so the
    model is first compared against the DUT outputs and then ticked with the
    same inputs.
    """

    def build_phase(self):
        self.dut = cocotb.top
        self.model = CycleModel(ROWS, COLS, PIPE)
        self.cycles = 0
        self.mismatches = 0

    async def run_phase(self):
        dut = self.dut
        while True:
            await RisingEdge(dut.clk)
            if not bit(dut.rst_n):
                self.model.reset()
                continue
            exp = self.model.outputs()
            got = {
                "in_ready": bit(dut.in_ready),
                "out_valid": bit(dut.out_valid),
                "out_last": bit(dut.out_last),
                "out_row": unpack_int32(int(dut.out_row.value), COLS),
            }
            if got != exp:
                self.mismatches += 1
                if self.mismatches <= 5:
                    self.logger.error(f"cycle {cycle_now()}: DUT {got} != model {exp}")
            in_valid = bit(dut.in_valid)
            self.model.tick(
                in_valid,
                unpack_int8(int(dut.in_a.value), ROWS),
                unpack_int8(int(dut.in_b.value), COLS),
                bit(dut.in_last),
                bit(dut.out_ready),
            )
            self.cycles += 1


class Coverage(uvm_subscriber):
    """Functional coverage on what the DUT actually received."""

    def build_phase(self):
        bins = ["K=1", "K=2..ROWS-1", "K=ROWS..4N", "K>4N",
                "data: -128*-128", "data: 127*127", "data: -128*127", "data: all zero",
                "stall: input bubble mid-job", "stall: output backpressure",
                "flow: back-to-back jobs",
                "cross: K>4N with input bubble", "cross: K=1 back-to-back"]
        if ROWS <= 2:
            bins.remove("K=2..ROWS-1")
        self.hits = {b: 0 for b in bins}

    def _hit(self, name):
        if name in self.hits:
            self.hits[name] += 1

    def write(self, job):
        a, b = job["a"], job["b"]
        k = a.shape[1]
        big = max(ROWS, COLS)
        if k == 1:
            self._hit("K=1")
        elif k < ROWS:
            self._hit("K=2..ROWS-1")
        elif k <= 4 * big:
            self._hit("K=ROWS..4N")
        else:
            self._hit("K>4N")
        # A product a[r,k]*b[k,c] exists for every r, c, so a value pair is
        # exercised when it appears in the same column of A / row of B.
        for kk in range(k):
            col, row = a[:, kk], b[kk, :]
            if (col == -128).any() and (row == -128).any():
                self._hit("data: -128*-128")
            if (col == 127).any() and (row == 127).any():
                self._hit("data: 127*127")
            if ((col == -128).any() and (row == 127).any()) or ((col == 127).any() and (row == -128).any()):
                self._hit("data: -128*127")
        if not a.any() and not b.any():
            self._hit("data: all zero")
        if job["in_gap"]:
            self._hit("stall: input bubble mid-job")
            if k > 4 * big:
                self._hit("cross: K>4N with input bubble")
        if job["back_to_back"]:
            self._hit("flow: back-to-back jobs")
            if k == 1:
                self._hit("cross: K=1 back-to-back")

    def note_backpressure(self, res):
        if res["backpressure"]:
            self._hit("stall: output backpressure")

    def percent(self) -> float:
        return 100.0 * sum(1 for v in self.hits.values() if v) / len(self.hits)

    def report_phase(self):
        self.logger.info(f"Functional coverage: {self.percent():.1f} %")
        for name, n in self.hits.items():
            self.logger.info(f"  {'HIT ' if n else 'MISS'} {name:36s} {n}")


class _BackpressureTap(uvm_subscriber):
    def build_phase(self):
        self.cov = None

    def write(self, res):
        self.cov.note_backpressure(res)


# ---------------------------------------------------------------------------
# Environment and tests
# ---------------------------------------------------------------------------
class SaEnv(uvm_env):
    def build_phase(self):
        self.in_agent = InAgent("in_agent", self)
        self.out_agent = OutAgent("out_agent", self)
        self.scoreboard = Scoreboard("scoreboard", self)
        self.lockstep = LockstepChecker("lockstep", self)
        self.coverage = Coverage("coverage", self)
        self.bp_tap = _BackpressureTap("bp_tap", self)

    def connect_phase(self):
        self.in_agent.monitor.ap.connect(self.scoreboard.in_export)
        self.out_agent.monitor.ap.connect(self.scoreboard.out_export)
        self.in_agent.monitor.ap.connect(self.coverage.analysis_export)
        self.out_agent.monitor.ap.connect(self.bp_tap.analysis_export)
        self.bp_tap.cov = self.coverage


class BaseTest(uvm_test):
    gap_prob = GAP_PROB
    bp_prob = BP_PROB
    seq_cls = RandomSeq
    n_jobs = N_JOBS
    need_full_coverage = False
    check_latency = False

    def build_phase(self):
        ConfigDB().set(None, "*", "GAP_PROB", self.gap_prob)
        ConfigDB().set(None, "*", "BP_PROB", self.bp_prob)
        self.env = SaEnv("env", self)

    def start_of_simulation_phase(self):
        dut = cocotb.top
        for sig in (dut.in_valid, dut.in_a, dut.in_b, dut.in_last, dut.out_ready):
            sig.value = 0
        dut.rst_n.value = 0
        Clock(dut.clk, CLK_NS, unit="ns").start()

    async def run_phase(self):
        self.raise_objection()
        dut = cocotb.top
        for _ in range(5):
            await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        dut.rst_n.value = 1

        seq = self.seq_cls("seq")
        await seq.start(self.env.in_agent.seqr)

        # Wait for the last results to come out (bounded).
        mon = self.env.out_agent.monitor
        for _ in range(100_000):
            if mon.jobs_seen >= self.n_jobs:
                break
            await RisingEdge(dut.clk)
        for _ in range(5):
            await RisingEdge(dut.clk)
        self.drop_objection()

    def check_phase(self):
        sb, ls, cov = self.env.scoreboard, self.env.lockstep, self.env.coverage
        dut_errors = int(cocotb.top.check_errors.value)
        self.logger.info(
            f"[{ROWS}x{COLS} pipe={PIPE}] jobs passed {sb.passed}/{self.n_jobs}, failed {sb.failed}; "
            f"lockstep cycles {ls.cycles}, mismatches {ls.mismatches}; "
            f"RTL assertion failures {dut_errors}; coverage {cov.percent():.1f} %")
        assert sb.failed == 0, "scoreboard mismatches"
        assert sb.passed == self.n_jobs, f"only {sb.passed} of {self.n_jobs} jobs completed"
        assert ls.mismatches == 0, "cycle-accurate model disagrees with RTL"
        assert dut_errors == 0, "RTL assertions fired"
        if self.need_full_coverage:
            assert cov.percent() == 100.0, "functional coverage not closed"
        if self.check_latency:
            overhead = (ROWS + COLS - 2 + PIPE) + ROWS
            bad = [(k, got) for k, got, stalled in sb.latencies if not stalled and got != k + overhead]
            assert not bad, f"latency differs from K+{overhead} for {bad[:3]}"
            self.logger.info(f"latency check: {len(sb.latencies)} jobs, all equal K+{overhead}")


@pyuvm.test()
class DirectedTest(BaseTest):
    """Corner cases, no stalls, exact latency check."""
    gap_prob = 0.0
    bp_prob = 0.0
    seq_cls = DirectedSeq
    n_jobs = 12
    check_latency = True


@pyuvm.test()
class RandomStallTest(BaseTest):
    """Constrained-random jobs with random input bubbles and output backpressure."""
    need_full_coverage = True


# ---------------------------------------------------------------------------
# Reset in the middle of a job (plain cocotb test, no UVM env)
# ---------------------------------------------------------------------------
async def _drive_job(dut, a, b, stop_after_beats=None):
    k = a.shape[1]
    for kk in range(k):
        if stop_after_beats is not None and kk >= stop_after_beats:
            break
        dut.in_valid.value = 1
        dut.in_a.value = pack_int8(a[:, kk])
        dut.in_b.value = pack_int8(b[kk, :])
        dut.in_last.value = int(kk == k - 1)
        while True:
            await RisingEdge(dut.clk)
            if bit(dut.in_ready):
                break
    dut.in_valid.value = 0
    dut.in_last.value = 0


async def _collect(dut):
    rows = []
    while True:
        await RisingEdge(dut.clk)
        if bit(dut.out_valid) and bit(dut.out_ready):
            rows.append(unpack_int32(int(dut.out_row.value), COLS))
            if bit(dut.out_last):
                return np.array(rows, dtype=np.int64)


@cocotb.test()
async def reset_mid_job(dut):
    """Assert reset during FLUSH and during DRAIN, then check the next job is clean."""
    rng = np.random.default_rng(SEED)
    for sig in (dut.in_valid, dut.in_a, dut.in_b, dut.in_last):
        sig.value = 0
    dut.out_ready.value = 1
    dut.rst_n.value = 0
    Clock(dut.clk, CLK_NS, unit="ns").start()
    await Timer(5 * CLK_NS, unit="ns")
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1

    for phase in ("flush", "drain", "load"):
        a = rng.integers(-128, 128, (ROWS, 20))
        b = rng.integers(-128, 128, (20, COLS))
        if phase == "load":
            await _drive_job(dut, a, b, stop_after_beats=7)      # abandon mid-LOAD
        else:
            dut.out_ready.value = 0 if phase == "drain" else 1
            await _drive_job(dut, a, b)
            if phase == "drain":
                for _ in range(ROWS + COLS + 3):                # sit in DRAIN, stalled
                    await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        dut.rst_n.value = 0                                      # async reset mid-job
        await Timer(3 * CLK_NS, unit="ns")
        await FallingEdge(dut.clk)
        dut.rst_n.value = 1
        dut.out_ready.value = 1

        a2 = rng.integers(-128, 128, (ROWS, 9))
        b2 = rng.integers(-128, 128, (9, COLS))
        collector = cocotb.start_soon(_collect(dut))
        await _drive_job(dut, a2, b2)
        got = await collector
        exp = reference_matmul(a2, b2)
        assert np.array_equal(got, exp), f"wrong result after reset in {phase}: {got} vs {exp}"
        dut._log.info(f"reset during {phase.upper()}: next job correct")
    assert int(dut.check_errors.value) == 0, "RTL assertions fired"
