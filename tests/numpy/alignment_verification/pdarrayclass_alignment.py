import operator

from typing import Callable, Tuple

import numpy as np
import pytest

import arkouda as ak


# -----------------------------
# Helpers
# -----------------------------
def _np_dtype_for_kind(kind: str) -> np.dtype:
    if kind == "int":
        return np.dtype(np.int64)
    if kind == "uint":
        return np.dtype(np.uint64)
    if kind == "float":
        return np.dtype(np.float64)
    if kind == "bool":
        return np.dtype(np.bool_)
    raise ValueError(f"unknown kind={kind}")


def _make_data(kind: str, n: int, seed: int = 0) -> Tuple[np.ndarray, "ak.pdarray"]:
    rng = np.random.default_rng(seed)
    dt = _np_dtype_for_kind(kind)

    if kind == "int":
        a_np = rng.integers(-100, 100, size=n, dtype=dt)
        a_ak = ak.array(a_np)
        return a_np, a_ak

    if kind == "uint":
        a_np = rng.integers(0, 200, size=n, dtype=dt)
        a_ak = ak.array(a_np)
        return a_np, a_ak

    if kind == "float":
        a_np = rng.normal(size=n).astype(dt)
        # sprinkle NaNs to exercise NaN semantics
        if n >= 10:
            a_np[::10] = np.nan
        a_ak = ak.array(a_np)
        return a_np, a_ak

    if kind == "bool":
        a_np = rng.integers(0, 2, size=n, dtype=np.int8).astype(dt)
        a_ak = ak.array(a_np)
        return a_np, a_ak

    raise ValueError(f"unknown kind={kind}")


def _assert_np_ak_same(a_np: np.ndarray, a_ak: "ak.pdarray") -> None:
    got = a_ak.to_ndarray()

    assert got.shape == a_np.shape

    # dtype alignment is sometimes intentionally different (e.g. int32 vs int64),
    # but pdarray typically uses 64-bit types; adjust this if your project differs.
    # This checks "kind" alignment rather than exact dtype string.
    assert got.dtype.kind == a_np.dtype.kind

    if got.dtype.kind == "f":
        np.testing.assert_allclose(got, a_np, equal_nan=True, rtol=1e-12, atol=0.0)
    else:
        np.testing.assert_array_equal(got, a_np)


# -----------------------------
# Binary operator alignment
# -----------------------------
_BINARY_CASES = [
    # (op_name, numpy_callable, python_operator_callable)
    ("add", np.add, operator.add),
    ("sub", np.subtract, operator.sub),
    ("mul", np.multiply, operator.mul),
    ("truediv", np.true_divide, operator.truediv),
    ("floordiv", np.floor_divide, operator.floordiv),
    ("mod", np.mod, operator.mod),
    ("pow", np.power, operator.pow),
    ("and", np.bitwise_and, operator.and_),
    ("or", np.bitwise_or, operator.or_),
    ("xor", np.bitwise_xor, operator.xor),
    ("lshift", np.left_shift, operator.lshift),
    ("rshift", np.right_shift, operator.rshift),
]


def _binary_op_supported(kind: str, opname: str) -> bool:
    if kind == "bool":
        return opname in {
            "add",  # +
            "mul",  # *
            "pow",  # **
            "and",  # &
            "or",  # |
            "xor",  # ^
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
        }

    if opname in {"lshift", "rshift"}:
        return kind in {"int", "uint"}

    if opname in {"and", "or", "xor"}:
        return kind in {"int", "uint", "bool"}

    return True


@pytest.mark.parametrize("kind", ["int", "uint", "float", "bool"])
@pytest.mark.parametrize("opname,np_op,py_op", _BINARY_CASES)
def test_pdarray_binary_ops_match_numpy(
    kind: str, opname: str, np_op: Callable, py_op: Callable
) -> None:
    if not _binary_op_supported(kind, opname):
        pytest.skip(f"{opname} not supported for {kind}")

    # --- Known NumPy alignment gaps (intentional xfails) ---

    # 1) Signed integer floor-division semantics
    # NumPy: floor toward -inf
    # Arkouda: truncation toward 0
    if kind == "int" and opname == "floordiv":
        pytest.xfail("Arkouda uses truncating division for signed ints; NumPy uses floor division")

    # 2) Signed integer modulo semantics (tied to floor-division)
    # NumPy: remainder has sign of divisor
    # Arkouda: remainder consistent with trunc division
    if kind == "int" and opname == "mod":
        pytest.xfail(
            "Arkouda modulo follows truncating division; NumPy remainder follows floor-division rules"
        )

    # 3) Signed integer right shift
    # NumPy: arithmetic right shift (sign-propagating)
    # Arkouda: logical / zero-fill right shift
    if kind == "int" and opname == "rshift":
        pytest.xfail("Arkouda right shift on signed ints is logical; NumPy uses arithmetic shift")

    # 4) Boolean power dtype promotion
    # NumPy: bool ** bool -> signed int
    # Arkouda: returns unsigned
    if kind == "bool" and opname == "pow":
        pytest.xfail("Arkouda bool ** bool returns unsigned dtype; NumPy promotes to signed int")

    n = 101
    a_np, a_ak = _make_data(kind, n, seed=1)
    b_np, b_ak = _make_data(kind, n, seed=2)

    # Avoid division/mod by zero instability
    if opname in {"truediv", "floordiv", "mod"}:
        if kind in {"int", "uint"}:
            b_np = b_np.copy()
            b_np[b_np == 0] = 1
            b_ak = ak.array(b_np)
        elif kind == "float":
            b_np = b_np.copy()
            b_np[np.isnan(b_np)] = 1.0
            b_np[b_np == 0.0] = 1.0
            b_ak = ak.array(b_np)

    # Avoid huge pow overflow for ints
    if opname == "pow" and kind in {"int", "uint"}:
        a_np = (a_np % 10).astype(a_np.dtype)
        b_np = (np.abs(b_np) % 5).astype(b_np.dtype)
        a_ak = ak.array(a_np)
        b_ak = ak.array(b_np)

    # pdarray OP pdarray
    got_ak = py_op(a_ak, b_ak)
    got_np = np_op(a_np, b_np)

    _assert_np_ak_same(got_np, got_ak)

    # pdarray OP scalar
    scalar = 3
    if kind == "float":
        scalar = 3.5

    # For shifts, scalar must be non-negative and small
    if opname in {"lshift", "rshift"}:
        scalar = 2

    got_ak2 = py_op(a_ak, scalar)
    got_np2 = np_op(a_np, scalar)
    _assert_np_ak_same(got_np2, got_ak2)

    # scalar OP pdarray (reverse op)
    got_ak3 = py_op(scalar, a_ak)
    got_np3 = np_op(scalar, a_np)
    _assert_np_ak_same(got_np3, got_ak3)


# -----------------------------
# Comparisons
# -----------------------------
_COMPARE_CASES = [
    ("lt", np.less, operator.lt),
    ("le", np.less_equal, operator.le),
    ("gt", np.greater, operator.gt),
    ("ge", np.greater_equal, operator.ge),
    ("eq", np.equal, operator.eq),
    ("ne", np.not_equal, operator.ne),
]


@pytest.mark.parametrize("kind", ["int", "uint", "float", "bool"])
@pytest.mark.parametrize("opname,np_op,py_op", _COMPARE_CASES)
def test_pdarray_comparisons_match_numpy(
    kind: str, opname: str, np_op: Callable, py_op: Callable
) -> None:
    n = 97
    a_np, a_ak = _make_data(kind, n, seed=11)
    b_np, b_ak = _make_data(kind, n, seed=12)

    got_ak = py_op(a_ak, b_ak)
    got_np = np_op(a_np, b_np)

    # comparisons should produce bool arrays
    assert got_ak.dtype == ak.bool_
    np.testing.assert_array_equal(got_ak.to_ndarray(), got_np)


# -----------------------------
# Unary ops
# -----------------------------
_UNARY_CASES = [
    ("neg", np.negative, operator.neg),
    ("pos", np.positive, operator.pos),
    ("invert", np.invert, operator.invert),
]


def _unary_supported(kind: str, opname: str) -> bool:
    if opname == "invert":
        return kind in {"int", "uint", "bool"}
    return True


@pytest.mark.parametrize("kind", ["int", "uint", "float", "bool"])
@pytest.mark.parametrize("opname,np_op,py_op", _UNARY_CASES)
def test_pdarray_unary_ops_match_numpy(kind: str, opname: str, np_op: Callable, py_op: Callable) -> None:
    if not _unary_supported(kind, opname):
        pytest.skip(f"{opname} not supported for {kind}")

    a_np, a_ak = _make_data(kind, 123, seed=21)

    # If NumPy raises for this unary op/dtype, Arkouda should also raise.
    try:
        expected_np = np_op(a_np)  # noqa: F841
    except TypeError:
        with pytest.raises(TypeError):
            py_op(a_ak)
        return

    got_ak = py_op(a_ak)
    got_np = np_op(a_np)
    _assert_np_ak_same(got_np, got_ak)


# -----------------------------
# Indexing / slicing alignment
# -----------------------------
@pytest.mark.parametrize("kind", ["int", "float", "bool"])
def test_pdarray_basic_slicing_matches_numpy(kind: str) -> None:
    a_np, a_ak = _make_data(kind, 200, seed=31)

    slices = [
        slice(None, None, None),
        slice(0, 10, None),
        slice(5, 50, 3),
        slice(-50, None, None),
        slice(None, None, -1),
        slice(150, 20, -7),
    ]

    for s in slices:
        got_ak = a_ak[s]
        got_np = a_np[s]
        _assert_np_ak_same(got_np, got_ak)


@pytest.mark.parametrize("kind", ["int", "float"])
def test_pdarray_boolean_mask_indexing_matches_numpy(kind: str) -> None:
    a_np, a_ak = _make_data(kind, 120, seed=41)
    mask_np, mask_ak = _make_data("bool", 120, seed=42)

    got_ak = a_ak[mask_ak]
    got_np = a_np[mask_np]
    _assert_np_ak_same(got_np, got_ak)


# -----------------------------
# Reshape / flatten / take
# -----------------------------
@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("kind", ["int", "float", "bool"])
def test_pdarray_reshape_and_flatten_match_numpy(kind: str) -> None:
    a_np, a_ak = _make_data(kind, 240, seed=51)

    np_reshaped = a_np.reshape((16, 15))
    ak_reshaped = a_ak.reshape((16, 15))
    _assert_np_ak_same(np_reshaped, ak_reshaped)

    np_flat = np_reshaped.flatten()
    ak_flat = ak_reshaped.flatten()
    _assert_np_ak_same(np_flat, ak_flat)


@pytest.mark.parametrize("kind", ["int", "float"])
def test_pdarray_take_matches_numpy(kind: str) -> None:
    a_np, a_ak = _make_data(kind, 100, seed=61)

    idx_np = np.array([0, 3, 3, 9, 50, 99], dtype=np.int64)
    idx_ak = ak.array(idx_np)

    got_ak = a_ak.take(idx_ak)
    got_np = np.take(a_np, idx_np)

    _assert_np_ak_same(got_np, got_ak)


# -----------------------------
# Misc "array contract" behaviors
# -----------------------------
def test_pdarray_len_matches_numpy() -> None:
    a_np, a_ak = _make_data("int", 37, seed=71)
    assert len(a_ak) == len(a_np)


def test_pdarray_bool_raises_like_numpy_for_non_scalar() -> None:
    # NumPy: bool(np.array([1,2])) raises ValueError: ambiguous truth value
    a_np = np.array([1, 2], dtype=np.int64)
    a_ak = ak.array(a_np)

    with pytest.raises(ValueError):
        bool(a_np)

    with pytest.raises(ValueError):
        bool(a_ak)


@pytest.mark.parametrize("kind", ["int", "float", "bool"])
def test_pdarray_equals_matches_numpy_array_equal(kind: str) -> None:
    a_np, a_ak = _make_data(kind, 55, seed=81)
    b_np = a_np.copy()
    b_ak = ak.array(b_np)

    assert a_ak.equals(b_ak) == np.array_equal(a_np, b_np)

    # mutate b
    b_np2 = b_np.copy()
    if kind == "float":
        b_np2[0] = 123.0
    else:
        b_np2[0] = ~b_np2[0] if kind == "bool" else (b_np2[0] + 1)
    b_ak2 = ak.array(b_np2)

    assert a_ak.equals(b_ak2) == np.array_equal(a_np, b_np2)


def test_helpers_raise_on_unknown_kind() -> None:
    with pytest.raises(ValueError, match=r"unknown kind="):
        _np_dtype_for_kind("nope")

    with pytest.raises(ValueError, match=r"unknown kind="):
        _make_data("nope", 10, seed=0)
