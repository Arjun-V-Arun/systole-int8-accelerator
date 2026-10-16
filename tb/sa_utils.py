"""Shared helpers: bit packing for the flat RTL ports and the NumPy reference."""

import numpy as np

DW = 8
ACC_W = 32


def pack_int8(values) -> int:
    """Pack signed INT8 values into one integer, element i at bits [i*8 +: 8]."""
    word = 0
    for i, v in enumerate(values):
        word |= (int(v) & 0xFF) << (DW * i)
    return word


def unpack_int8(word: int, n: int) -> list[int]:
    out = []
    for i in range(n):
        v = (word >> (DW * i)) & 0xFF
        out.append(v - 256 if v & 0x80 else v)
    return out


def unpack_int32(word: int, n: int) -> list[int]:
    out = []
    for i in range(n):
        v = (word >> (ACC_W * i)) & 0xFFFFFFFF
        out.append(v - (1 << 32) if v & 0x80000000 else v)
    return out


def reference_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """INT8 x INT8 matmul with a 32-bit two's complement accumulator.

    Computed in int64 and then wrapped, which matches the RTL even if a very
    long K ever overflowed 32 bits (it cannot for K < 131072).
    """
    c = a.astype(np.int64) @ b.astype(np.int64)
    return ((c + (1 << 31)) % (1 << 32) - (1 << 31)).astype(np.int64)
