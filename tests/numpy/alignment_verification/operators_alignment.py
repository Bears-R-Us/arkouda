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


# --- Parametrized binary op tests (array ⊗ array) ---
@pytest.mark.xfail(reason="Requires ak.minimum (#5118) and bug fix for rshift (#5115)")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,fn", BINARY_OPS)
def test_binary_ops_array_array_alignment(dtype, name, fn):
    if dtype is ak.bool_ and name in {"add", "sub", "mul", "truediv", "floordiv", "mod", "pow"}:
        pytest.skip(f"NumPy does not support {name} for boolean dtype")
    # Skip invalid dtype/op combinations
    integer_only = name in {"lshift", "rshift", "and_", "or_", "xor"}
    if integer_only and dtype is ak.float64:
        pytest.skip(f"{name} not valid for float64")

    if name in {"lshift", "rshift"} and dtype is ak.bool_:
        pytest.skip(f"{name} not meaningful for bool")

    # Prepare operands with constraints to avoid undefined behavior
    rhs_nonzero = name in {"truediv", "floordiv", "mod"}
    rhs_nonneg = name in {"lshift", "rshift", "pow"}

    a = _rand_array(dtype)
    b = _rand_array(dtype, nonzero=rhs_nonzero, nonneg=rhs_nonneg)

    # Extra guarding for pow to keep magnitudes reasonable
    if name == "pow":
        if dtype is ak.float64:
            b = ak.minimum(b, 6.0)  # floats: reasonable exponent

        elif dtype in (ak.int64, ak.uint64):
            b = ak.minimum(b, 7)  # ints: small exponent to prevent overflow stalls

        elif dtype is ak.bool_:
            # Boolean exponents are {0,1}
            b = b % 2

    ak_res = fn(a, b)
    np_res = fn(_numpy_equivalent(a), _numpy_equivalent(b))

    # Division on ints yields float in NumPy; Arkouda should match
    if name == "truediv" or (dtype is ak.float64):
        assert_almost_equivalent(ak_res, np_res)
    else:
        assert_arkouda_array_equivalent(ak_res, np_res)


# --- Scalar broadcasting (array ⊗ scalar and scalar ⊗ array) ---
@pytest.mark.xfail(reason="Requires bug fixes for floordiv, mod, pow (#5113, #5112, #5114)")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,fn", BINARY_OPS)
def test_binary_ops_array_scalar_alignment(dtype, name, fn):
    # Arithmetic on booleans is not supported by NumPy
    if dtype is ak.bool_ and name in {"add", "sub", "mul", "truediv", "floordiv", "mod", "pow"}:
        pytest.skip(f"NumPy does not support {name} for boolean dtype")

    integer_only = name in {"lshift", "rshift", "and_", "or_", "xor"}
    if integer_only and dtype is ak.float64:
        pytest.skip(f"{name} not valid for float64")
    if name in {"lshift", "rshift"} and dtype is ak.bool_:
        pytest.skip(f"{name} not meaningful for bool")

    a = _rand_array(dtype)

    # Choose a scalar compatible with dtype and operator constraints
    if dtype is ak.bool_:
        scalar = True
    elif dtype is ak.float64:
        scalar = 3.5
    elif dtype is ak.uint64:
        scalar = np.uint64(5)
    else:  # int64
        scalar = 5

    if name in {"truediv", "floordiv", "mod"}:
        # Ensure scalar is nonzero when used as RHS
        scalar_rhs = scalar if (scalar != 0) else (1 if dtype is not ak.float64 else 1.0)
    else:
        scalar_rhs = scalar

    # power/shift require nonnegative RHS
    if name in {"lshift", "rshift", "pow"}:
        scalar_rhs = abs(scalar_rhs)

    # array OP scalar
    ak_res = fn(a, scalar_rhs)
    np_res = fn(_numpy_equivalent(a), scalar_rhs)
    if name == "truediv" or (dtype is ak.float64):
        assert_almost_equivalent(ak_res, np_res)
    else:
        assert_arkouda_array_equivalent(ak_res, np_res)

    # scalar OP array
    ak_res2 = fn(scalar_rhs, a)
    np_res2 = fn(scalar_rhs, _numpy_equivalent(a))
    if name == "truediv" or (dtype is ak.float64):
        assert_almost_equivalent(ak_res2, np_res2)
    else:
        assert_arkouda_array_equivalent(ak_res2, np_res2)


# --- Comparisons (array ⊗ array) ---
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,fn", CMP_OPS)
def test_comparison_ops_alignment(dtype, name, fn):
    a = _rand_array(dtype)
    b = _rand_array(dtype)
    ak_res = fn(a, b)
    np_res = fn(_numpy_equivalent(a), _numpy_equivalent(b))
    assert_arkouda_array_equivalent(ak_res, np_res)


# --- Unary operator alignment ---
@pytest.mark.xfail(reason="Requires pdarray.__pos__ (#5116) and bug fixes in __neg__ (#5117, #5119)")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,fn", UNARY_OPS)
def test_unary_ops_alignment(dtype, name, fn):
    if name == "invert" and dtype is ak.float64:
        pytest.skip("bitwise invert not defined for float64")

    a = _rand_array(dtype, nonneg=(name == "invert"))  # for invert on uint shifts OK either way

    # Special-case: boolean negation should raise TypeError in both Arkouda and NumPy
    if name == "neg" and dtype is ak.bool_:
        with pytest.raises(TypeError):
            fn(a)  # Arkouda: -a
        with pytest.raises(TypeError):
            fn(_numpy_equivalent(a))  # NumPy: -array(...)
        return

    ak_res = fn(a)
    np_res = fn(_numpy_equivalent(a))

    # float neg/pos use almost equivalent
    if dtype is ak.float64:
        assert_almost_equivalent(ak_res, np_res)
    else:
        assert_arkouda_array_equivalent(ak_res, np_res)
