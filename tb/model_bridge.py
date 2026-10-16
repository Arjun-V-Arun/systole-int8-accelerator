"""ctypes bridge to the cycle-accurate C++ model (model/libsamodel.so)."""

import ctypes
import os
from pathlib import Path

_LIB_PATH = Path(os.environ.get(
    "SA_MODEL_LIB",
    Path(__file__).resolve().parent.parent / "model" / "libsamodel.so",
))

_lib = ctypes.CDLL(str(_LIB_PATH))
_lib.sa_new.restype = ctypes.c_void_p
_lib.sa_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.sa_free.argtypes = [ctypes.c_void_p]
_lib.sa_reset.argtypes = [ctypes.c_void_p]
_lib.sa_outputs.argtypes = [ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int32)]
_lib.sa_tick.argtypes = [ctypes.c_void_p, ctypes.c_int,
                         ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8),
                         ctypes.c_int, ctypes.c_int]
_lib.sa_counters.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4


class CycleModel:
    """Python handle around sa::Model. Same semantics as the RTL ports."""

    def __init__(self, rows: int, cols: int, pipe_mac: int = 0):
        self.rows, self.cols = rows, cols
        self._h = _lib.sa_new(rows, cols, pipe_mac)
        self._a = (ctypes.c_int8 * rows)()
        self._b = (ctypes.c_int8 * cols)()
        self._row = (ctypes.c_int32 * cols)()

    def __del__(self):
        if getattr(self, "_h", None):
            _lib.sa_free(self._h)
            self._h = None

    def reset(self):
        _lib.sa_reset(self._h)

    def outputs(self) -> dict:
        ir, ov, ol = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        _lib.sa_outputs(self._h, ctypes.byref(ir), ctypes.byref(ov), ctypes.byref(ol), self._row)
        return {"in_ready": ir.value, "out_valid": ov.value, "out_last": ol.value,
                "out_row": list(self._row)}

    def tick(self, in_valid: int, in_a: list[int], in_b: list[int], in_last: int, out_ready: int):
        for i, v in enumerate(in_a):
            self._a[i] = v
        for i, v in enumerate(in_b):
            self._b[i] = v
        _lib.sa_tick(self._h, in_valid, self._a, self._b, in_last, out_ready)

    def counters(self) -> dict:
        vals = [ctypes.c_uint64() for _ in range(4)]
        _lib.sa_counters(self._h, *[ctypes.byref(v) for v in vals])
        return dict(zip(["cycles", "steps", "useful_macs", "jobs"], [v.value for v in vals]))
