"""
NumPy alignment tests for Arkouda array-creation APIs.

Scope: functions implemented in arkouda.numpy.pdarraycreation
(array/zeros/ones/full/like/arange/linspace/logspace).
These tests compare Arkouda behavior to NumPy where a like-for-like comparison makes sense, and
explicitly document intentional differences
(e.g., deep-copy semantics, Strings handling, bigint support).

Run:
    pytest -q pdarraycreation_alignment.py

Notes
-----
- These tests require an Arkouda server connection (ak.connect() in your conftest/fixtures).
- If your repo already provides an `ak_server` fixture, these tests will use it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import arkouda as ak


# -------------------------
# Helpers
# -------------------------


def _np_dtype_name(dt: Any) -> str:
    """Normalize dtype into a readable name."""
    if dt is None:
        return "None"
    try:
        return np.dtype(dt).name
    except Exception:
        return str(dt)


def _as_np(x: Any) -> np.ndarray:
    """Convert Arkouda object to numpy ndarray."""
    if hasattr(x, "to_ndarray"):
        return x.to_ndarray()
    return np.asarray(x)


def assert_np_equal(
    got: np.ndarray,
    exp: np.ndarray,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    """Assert equality with NaN-aware comparison for floats."""
    got = np.asarray(got)
    exp = np.asarray(exp)

    assert got.shape == exp.shape, f"shape mismatch: got={got.shape} exp={exp.shape}"
    assert got.dtype == exp.dtype, f"dtype mismatch: got={got.dtype} exp={exp.dtype}"

    if np.issubdtype(got.dtype, np.floating):
        # numpy has assert_allclose that is NaN-aware with equal_nan=True
        np.testing.assert_allclose(got, exp, rtol=rtol, atol=atol, equal_nan=True)
    else:
        np.testing.assert_array_equal(got, exp)


def assert_ak_matches_np(
    ak_obj: Any,
    np_obj: Any,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    """Compare Arkouda result with NumPy result by value, dtype, and shape."""
    got = _as_np(ak_obj)
    exp = np.asarray(np_obj)

    # For strings, NumPy dtype can be '<U...' while Arkouda returns object/str in ndarray;
    # normalize by casting both to object for comparison.
    if got.dtype.kind in ("U", "S") or exp.dtype.kind in ("U", "S"):
        got = got.astype(object)
        exp = exp.astype(object)

    # dtype alignment expectations:
    # - numeric dtypes should match exactly
    # - strings compared as object arrays
    if got.dtype.kind not in ("O",) and exp.dtype.kind not in ("O",):
        assert got.dtype == exp.dtype, f"dtype mismatch: got={got.dtype} exp={exp.dtype}"

    assert got.shape == exp.shape, f"shape mismatch: got={got.shape} exp={exp.shape}"

    if got.dtype.kind in ("f",) and exp.dtype.kind in ("f",):
        np.testing.assert_allclose(got, exp, rtol=rtol, atol=atol, equal_nan=True)
    else:
        np.testing.assert_array_equal(got, exp)


# -------------------------
# ak.array alignment
# -------------------------


@pytest.mark.parametrize(
    "a",
    [
        [],
        [1, 2, 3],
        [1.5, 2.5, np.nan],
        np.arange(10),
        np.array([True, False, True]),
        np.array(["a", "bb", "ccc"], dtype="U"),
        (i for i in range(5)),  # generator -> list
    ],
)
@pytest.mark.parametrize("dtype", [None, np.int64, np.float64, np.bool_, "int64", "float64"])
def test_array_basic_matches_numpy(a, dtype):
    np_arr = np.array(
        list(a) if hasattr(a, "__iter__") and not isinstance(a, (list, tuple, np.ndarray)) else a
    )

    if dtype is None:
        exp = np.array(np_arr)
        got = ak.array(np_arr)
        assert_ak_matches_np(got, exp)
        return

    # If NumPy errors, Arkouda should error with the same exception type.
    try:
        exp = np.array(np_arr, dtype=dtype)
    except Exception as np_exc:
        with pytest.raises(type(np_exc)):
            ak.array(np_arr, dtype=dtype)
        return

    # If NumPy succeeds, Arkouda must match.
    got = ak.array(np_arr, dtype=dtype)
    assert_ak_matches_np(got, exp)


def test_array_copy_semantics_pdarray():
    x = ak.arange(5)
    y = ak.array(x, copy=False)
    z = ak.array(x, copy=True)

    # copy=False returns same object (NumPy would be view-ish depending; Arkouda documents no views)
    assert y is x
    # copy=True deep-copies
    assert z is not x
    assert_ak_matches_np(z, x.to_ndarray())


def test_array_scalar_string_rejected():
    with pytest.raises(TypeError):
        ak.array("hello")  # Arkouda rejects scalar str; NumPy accepts and makes 0-d array


# -------------------------
# zeros/ones/full alignment
# -------------------------


@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("shape", [0, 1, 5, (0,), (2, 3)])
@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.bool_])
def test_zeros_matches_numpy(shape, dtype):
    got = ak.zeros(shape, dtype=dtype)
    exp = np.zeros(shape, dtype=dtype)
    assert_ak_matches_np(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("shape", [0, 1, 5, (0,), (2, 3)])
@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.bool_])
def test_ones_matches_numpy(shape, dtype):
    got = ak.ones(shape, dtype=dtype)
    exp = np.ones(shape, dtype=dtype)
    assert_ak_matches_np(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize(
    "shape,fill,dtype",
    [
        (5, 7, np.int64),
        ((2, 3), 9, np.float64),
        ((2, 3), True, np.bool_),
        ((0,), 11, np.int64),
    ],
)
def test_full_matches_numpy_numeric(shape, fill, dtype):
    got = ak.full(shape, fill, dtype=dtype)
    exp = np.full(shape, fill, dtype=dtype)
    assert_ak_matches_np(got, exp)


@pytest.mark.parametrize("shape,fill", [(0, "x"), (5, "hi"), (3, "🔥")])
def test_full_strings_matches_numpy_object(shape, fill):
    # Temporary: Arkouda currently corrupts non-BMP code points (e.g. emoji)
    if any(ord(ch) > 0xFFFF for ch in fill):
        pytest.xfail(
            "Known issue: non-BMP Unicode (emoji) corrupted in ak.Strings/ak.full. Issue #5266."
        )

    got = ak.full(shape, fill)
    exp = np.full(shape, fill, dtype=object)
    assert_ak_matches_np(got, exp)


# -------------------------
# *_like alignment
# -------------------------


@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.bool_])
def test_zeros_like_matches_numpy(dtype):
    if dtype is np.bool_:
        base_np = (np.arange(6) % 2).astype(np.bool_).reshape(2, 3)
    else:
        base_np = np.arange(6, dtype=dtype).reshape(2, 3)

    base_ak = ak.array(base_np)

    got = ak.zeros_like(base_ak)
    exp = np.zeros_like(base_np)
    assert_ak_matches_np(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.bool_])
def test_ones_like_matches_numpy(dtype):
    if dtype is np.bool_:
        base_np = (np.arange(6) % 2).astype(np.bool_).reshape(2, 3)
    else:
        base_np = np.arange(6, dtype=dtype).reshape(2, 3)

    base_ak = ak.array(base_np)

    got = ak.ones_like(base_ak)
    exp = np.ones_like(base_np)
    assert_ak_matches_np(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2])
def test_full_like_matches_numpy():
    base_np = np.arange(6, dtype=np.int64).reshape(2, 3)
    base_ak = ak.array(base_np)

    got = ak.full_like(base_ak, 42)
    exp = np.full_like(base_np, 42)
    assert_ak_matches_np(got, exp)


# -------------------------
# arange alignment
# -------------------------


@pytest.mark.parametrize(
    "args",
    [
        (0,),
        (5,),
        (0, 5),
        (2, 10),
        (0, 10, 2),
        (10, 0, -2),
        (-5, -10, -1),
        (5, 5, 1),
        (5, 5, -1),
    ],
)
@pytest.mark.parametrize("dtype", [None, np.int64, np.float64])
def test_arange_matches_numpy(args, dtype):
    # Arkouda arange always creates int64 then casts, so float64 is allowed.
    if dtype is None:
        got = ak.arange(*args)
        exp = np.arange(*args, dtype=np.int64)
    else:
        got = ak.arange(*args, dtype=dtype)
        exp = np.arange(*args, dtype=dtype)
    assert_ak_matches_np(got, exp)


def test_arange_step_zero_raises_like_numpy():
    with pytest.raises(ZeroDivisionError):
        ak.arange(0, 10, 0)


# -------------------------
# linspace/logspace alignment
# -------------------------


@pytest.mark.parametrize(
    "start,stop,num,endpoint",
    [
        (0, 1, 1, True),
        (0, 1, 2, True),
        (0, 1, 3, True),
        (0, 1, 3, False),
        (1, 0, 3, True),
        (-1.5, 2.5, 7, True),
    ],
)
def test_linspace_scalar_matches_numpy(start, stop, num, endpoint):
    if num == 1:
        pytest.xfail(
            "Known Arkouda bug: ak.linspace fails for num == 1 (ZeroDivisionError). Issue #5267."
        )

    got = ak.linspace(start, stop, num, endpoint=endpoint)
    exp = np.linspace(start, stop, num, endpoint=endpoint, dtype=np.float64)
    assert_ak_matches_np(got, exp)


def test_linspace_axis_scalar_rejected():
    # Arkouda raises ValueError if axis supplied for scalar start/stop and axis != 0
    with pytest.raises(ValueError):
        ak.linspace(0, 1, 5, axis=1)


@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("axis", [0, -1])
def test_linspace_vector_matches_numpy(axis):
    start_np = np.array([0.0, 1.0], dtype=np.float64)
    stop_np = np.array([2.0, 3.0], dtype=np.float64)

    start_ak = ak.array(start_np)
    stop_ak = ak.array(stop_np)

    got = ak.linspace(start_ak, stop_ak, 3, axis=axis)
    exp = np.linspace(start_np, stop_np, 3, axis=axis, dtype=np.float64)
    assert_ak_matches_np(got, exp)


@pytest.mark.parametrize(
    "start,stop,num,base,endpoint",
    [
        (0, 1, 3, 10.0, True),
        (2, 3, 3, 4.0, True),
        (0, 1, 3, 4.0, False),
        (1, 0, 3, 4.0, True),
    ],
)
def test_logspace_scalar_matches_numpy(start, stop, num, base, endpoint):
    got = ak.logspace(start, stop, num, base=base, endpoint=endpoint)
    exp = np.logspace(start, stop, num, base=base, endpoint=endpoint, dtype=np.float64)
    assert_ak_matches_np(got, exp)


def test_logspace_base_must_be_positive():
    with pytest.raises(ValueError):
        ak.logspace(0, 1, 5, base=0)


# -------------------------
# Documented differences / guardrails
# -------------------------


def test_zeros_empty_shape_tuple_not_supported_matches_doc():
    # Arkouda explicitly rejects size=() for zeros/full; NumPy supports it.
    with pytest.raises(ValueError):
        ak.zeros((), dtype=np.int64)


def test_full_empty_shape_tuple_not_supported_matches_doc():
    with pytest.raises(ValueError):
        ak.full((), 1, dtype=np.int64)


def test_np_dtype_name_helper():
    assert _np_dtype_name(None) == "None"
    assert _np_dtype_name(np.int64) == "int64"
    assert _np_dtype_name("not-a-dtype") == "not-a-dtype"


def test_as_np_fallback_to_numpy():
    x = [1, 2, 3]
    out = _as_np(x)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))


def test_assert_np_equal_non_float():
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([1, 2, 3], dtype=np.int64)
    assert_np_equal(a, b)
