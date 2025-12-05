import operator as op

from typing import Callable, List, Tuple

import numpy as np
import pytest

import arkouda as ak

from arkouda.testing import assert_almost_equivalent, assert_arkouda_array_equivalent


# --- Configuration ---
N = 100  # array length under test
SEED = 314159

# Dtypes to exercise; expand as Arkouda implements more ops for bigint/others.
DTYPES = [ak.int64, ak.uint64, ak.float64, ak.bool_]

# Binary operator matrix: (name, callable)
BINARY_OPS: List[Tuple[str, Callable]] = [
    ("add", op.add),
    ("sub", op.sub),
    ("mul", op.mul),
    ("truediv", op.truediv),
    ("floordiv", op.floordiv),
    ("mod", op.mod),
    ("pow", op.pow),
    ("lshift", op.lshift),
    ("rshift", op.rshift),
    ("and_", op.and_),
    ("or_", op.or_),
    ("xor", op.xor),
]

# Comparison operators (result dtype is boolean)
CMP_OPS: List[Tuple[str, Callable]] = [
    ("eq", op.eq),
    ("ne", op.ne),
    ("lt", op.lt),
    ("le", op.le),
    ("gt", op.gt),
    ("ge", op.ge),
]

# Unary ops
UNARY_OPS: List[Tuple[str, Callable]] = [
    ("pos", op.pos),
    ("neg", op.neg),
    ("invert", op.invert),  # bitwise not; invalid for float
]

# --- Helpers ---


def _rand_array(dtype, nonzero: bool = False, nonneg: bool = False):
    rng = np.random.default_rng(SEED)

    if dtype is ak.bool_:
        # Booleans from {0,1}
        data = rng.integers(0, 2, size=N, dtype=np.int64).astype(bool)
        return ak.array(data)

    if dtype is ak.float64:
        low, high = (-1000.0, 1000.0)
        data = rng.uniform(low, high, size=N)
        if nonzero:
            # push away from zero to avoid div-by-zero in denominator cases
            data = np.where(np.abs(data) < 1e-6, 1.0, data)
        if nonneg:
            data = np.abs(data)
        return ak.array(data)

    # Integer types
    if dtype is ak.uint64:
        low, high = (0, 1 << 31)
        data = rng.integers(low, high, size=N, dtype=np.uint64)
        if nonzero:
            data = np.where(data == 0, 1, data)
        if nonneg:
            # already nonnegative
            pass
        return ak.array(data)

    if dtype is ak.int64:
        low, high = (-1 << 31, 1 << 31)
        data = rng.integers(low, high, size=N, dtype=np.int64)
        if nonzero:
            data = np.where(data == 0, 1, data)
        if nonneg:
            data = np.abs(data)
        return ak.array(data)

    raise NotImplementedError(f"Unhandled dtype {dtype}")


def _numpy_equivalent(a: ak.pdarray):
    if isinstance(a, ak.pdarray):
        return a.to_ndarray()
    else:
        return a


# --- Unary operator alignment ---
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,fn", UNARY_OPS)
def test_unary_ops_alignment(dtype, name, fn):
    if name == "invert" and dtype is ak.float64:
        pytest.skip("bitwise invert not defined for float64")

    a = _rand_array(dtype, nonneg=(name == "invert"))  # for invert on uint shifts OK either way

    # Special-case: boolean negation and positive should raise on both sides
    if dtype is ak.bool_ and name in ("neg", "pos"):
        with pytest.raises(TypeError):
            fn(a)  # Arkouda: -a or +a
        with pytest.raises(TypeError):
            fn(_numpy_equivalent(a))  # NumPy: -array(...) or +array(...)
        return

    ak_res = fn(a)
    np_res = fn(_numpy_equivalent(a))

    # float neg/pos use almost equivalent
    if dtype is ak.float64:
        assert_almost_equivalent(ak_res, np_res)
    else:
        assert_arkouda_array_equivalent(ak_res, np_res)
