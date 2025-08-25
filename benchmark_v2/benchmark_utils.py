from builtins import sum as py_sum

import numpy as np
import pytest

import arkouda as ak


def calc_num_bytes(a) -> int:
    """
    Recursively calculate bytes for Arkouda arrays/strings and container types.
    Handles dicts, lists/tuples, and DataFrame-like objects (duck-typed).
    """
    # dict: sum values
    if isinstance(a, dict):
        return py_sum(calc_num_bytes(v) for v in a.values())

    # DataFrame-like (duck typing)
    if hasattr(a, "columns") and hasattr(a, "__getitem__"):
        try:
            return py_sum(calc_num_bytes(a[c]) for c in a.columns)
        except Exception:
            pass

    # Generic iterables (but not strings/bytes)
    if hasattr(a, "__iter__") and not isinstance(a, (str, bytes)):
        try:
            return py_sum(calc_num_bytes(v) for v in a)
        except Exception:
            pass

    # Arkouda arrays
    if isinstance(a, ak.pdarray):
        if a.dtype == "bigint":
            return a.size * 8 if 0 < pytest.max_bits <= 64 else a.size * 16
        else:
            return a.nbytes
    if isinstance(a, ak.Strings):
        num_bytes = a.get_bytes().nbytes + a.get_offsets().nbytes
        if hasattr(a, "entry"):
            num_bytes += a.entry.nbytes
        return num_bytes
    if isinstance(a, ak.Categorical):
        num_bytes = 0
        if hasattr(a, "codes"):
            num_bytes += a.codes.nbytes
        if hasattr(a, "categories"):
            num_bytes += a.categories.nbytes
        if hasattr(a, "segments"):
            num_bytes += a.segments.nbytes
        if hasattr(a, "permutation"):
            num_bytes += a.permutation.nbytes
        return num_bytes
    if isinstance(a, ak.SegArray):
        return a.values.nbytes + a.segments.nbytes
    if isinstance(a, np.ndarray):
        return a.nbytes

    raise TypeError(f"Unhandled type {type(a)} in calc_nbytes")
