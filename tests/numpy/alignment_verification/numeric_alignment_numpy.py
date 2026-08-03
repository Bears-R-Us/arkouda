"""
NumPy-alignment tests for arkouda numeric functions.

These tests are intended to catch behavioral drift versus NumPy for the functions
implemented in arkouda/numpy/numeric.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import arkouda as ak


# -------------------------
# Helpers
# -------------------------

RTOL = 1e-13
ATOL = 1e-13


def _to_np(x: Any) -> Any:
    """Convert Arkouda objects (pdarray/Strings) to NumPy/Python types."""
    if hasattr(x, "to_ndarray"):
        return x.to_ndarray()
    return x


def _assert_same_dtype_kind(np_res: np.ndarray, ak_res: Any) -> None:
    """Loose check: numeric result kinds should match (bool/int/uint/float)."""
    ak_np = _to_np(ak_res)
    if not isinstance(ak_np, np.ndarray):
        return
    # Allow NumPy to choose float64 in places Arkouda may upcast, but keep kind aligned.
    assert ak_np.dtype.kind == np_res.dtype.kind


def _assert_array_equal_or_allclose(np_res: np.ndarray, ak_res: Any, *, equal_nan: bool = True) -> None:
    ak_np = _to_np(ak_res)
    assert isinstance(ak_np, np.ndarray)
    assert ak_np.shape == np_res.shape

    if np_res.dtype.kind in ("f", "c"):
        np.testing.assert_allclose(ak_np, np_res, rtol=RTOL, atol=ATOL, equal_nan=equal_nan)
    else:
        np.testing.assert_array_equal(ak_np, np_res)


def _apply_numpy_where(x: np.ndarray, where: np.ndarray, np_func) -> np.ndarray:
    """
    Deterministic reference for Arkouda-style where semantics:
    apply ufunc where=True, otherwise keep the original x.
    """
    y = np_func(x)  # computes with correct result dtype (often float)
    return np.where(where, y, x)


def _ak_where_param(where: np.ndarray) -> Any:
    """Convert numpy boolean mask to Arkouda pdarray(bool)."""
    return ak.array(where.astype(bool))


# -------------------------
# Input generators
# -------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def base_int(rng: np.random.Generator) -> np.ndarray:
    x = rng.integers(-50, 50, size=200, dtype=np.int64)
    # ensure some zeros
    x[::37] = 0
    return x


@pytest.fixture(scope="module")
def base_uint(rng: np.random.Generator) -> np.ndarray:
    x = rng.integers(0, 100, size=200, dtype=np.uint64)
    x[::41] = 0
    return x


@pytest.fixture(scope="module")
def base_float(rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(loc=0.0, scale=3.0, size=200).astype(np.float64)
    # include special values
    x[0] = np.nan
    x[1] = np.inf
    x[2] = -np.inf
    x[::53] = 0.0
    return x


@pytest.fixture(scope="module")
def base_bool(rng: np.random.Generator) -> np.ndarray:
    x = rng.integers(0, 2, size=200, dtype=np.int64).astype(bool)
    return x


@pytest.fixture(scope="module")
def where_mask(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=200, dtype=np.int64).astype(bool)


# -------------------------
# Unary elementwise functions
# -------------------------

UNARY_FUNCS = [
    ("abs", np.abs, ak.abs, ("i", "u", "f")),
    ("fabs", np.fabs, ak.fabs, ("i", "u", "f")),  # Arkouda casts to float first
    ("ceil", np.ceil, ak.ceil, ("f",)),
    ("floor", np.floor, ak.floor, ("f",)),
    ("round", np.round, ak.round, ("f",)),
    ("trunc", np.trunc, ak.trunc, ("f",)),
    ("sign", np.sign, ak.sign, ("i", "f")),
    ("isfinite", np.isfinite, ak.isfinite, ("f", "i", "u", "b")),
    ("isinf", np.isinf, ak.isinf, ("f", "i", "u", "b")),
    ("isnan", np.isnan, ak.isnan, ("f", "i", "u", "b")),
    ("log", np.log, ak.log, ("f", "i", "u")),
    ("log2", np.log2, ak.log2, ("f", "i", "u")),
    ("log10", np.log10, ak.log10, ("f", "i", "u")),
    ("log1p", np.log1p, ak.log1p, ("f", "i", "u")),
    ("exp", np.exp, ak.exp, ("f", "i", "u")),
    ("expm1", np.expm1, ak.expm1, ("f", "i", "u")),
    ("square", np.square, ak.square, ("f", "i", "u", "b")),
    ("sin", np.sin, ak.sin, ("f", "i", "u")),
    ("cos", np.cos, ak.cos, ("f", "i", "u")),
    ("tan", np.tan, ak.tan, ("f", "i", "u")),
    ("arcsin", np.arcsin, ak.arcsin, ("f", "i", "u")),
    ("arccos", np.arccos, ak.arccos, ("f", "i", "u")),
    ("arctan", np.arctan, ak.arctan, ("f", "i", "u")),
    ("sinh", np.sinh, ak.sinh, ("f", "i", "u")),
    ("cosh", np.cosh, ak.cosh, ("f", "i", "u")),
    ("tanh", np.tanh, ak.tanh, ("f", "i", "u")),
    ("arcsinh", np.arcsinh, ak.arcsinh, ("f", "i", "u")),
    ("arccosh", np.arccosh, ak.arccosh, ("f", "i", "u")),
    ("arctanh", np.arctanh, ak.arctanh, ("f", "i", "u")),
]


@pytest.mark.parametrize("name,np_func,ak_func,kinds", UNARY_FUNCS)
@pytest.mark.parametrize("use_where", [False, True])
def test_unary_alignment(
    name: str,
    np_func,
    ak_func,
    kinds,
    use_where: bool,
    base_int,
    base_uint,
    base_float,
    base_bool,
    where_mask,
) -> None:
    # Suppress np warnings (divide by zero, invalid data, etc) for these tests.
    # We include invalid data to ensure that numpy and arkouda process it identically,
    # but it's not helpful having it clutter the output.

    old_settings = np.seterr(all="ignore")  # retrieve current settings
    np.seterr(over="ignore", invalid="ignore", divide="ignore")

    # Map dtype kinds to fixtures
    datasets = {
        "i": base_int,
        "u": base_uint,
        "f": base_float,
        "b": base_bool.astype(bool),
    }

    for kind in kinds:
        x = datasets[kind]

        # the general data in datasets[kind] may need adjustment depending on the function being tested.

        if name == "abs" and kind == "u":
            pytest.xfail(
                "Arkouda server does not support abs<uint64,1>; NumPy abs(uint) is identity. Issue #5247"
            )

        elif name == "isfinite" and kind in {"i", "u", "b"}:
            pytest.xfail(
                "ak.isfinite fails on non-float dtypes (NumPy returns all True for int/uint/bool); "
                "backend dispatch/casting bug: isfinite<1> cannot cast runtime types.  Issue #5248."
            )

        elif name == "isinf" and kind in {"i", "u", "b"}:
            pytest.xfail(
                "ak.isinf errors on non-float dtypes (NumPy returns all False for int/uint/bool); "
                "backend dispatch/casting bug: isinf<1> cannot cast runtime types. Issue #5249."
            )

        # The functions below have been rewritten to process "where" and "out" identically to numpy.
        # It is expected that as additional functions are rewritten, they will move from the above
        # test to this one.
        # Numpy functions have a way of returning unusual dtypes if the input is bool (sometimes
        # int8, sometimes int16).  Since arkouda doesn't handle such types, we convert bool
        # input to int64 below, so the tests will pass.

        elif use_where and name in {
            "ceil",
            "floor",
            "trunc",
            "square",
        }:
            np_where = where_mask
            np_out = np.ones_like(x if kind != "b" else x.astype(np.int64))  # handle the bool case
            np_res = np_func(x, np_out, where=np_where)  # this behavior differs from _apply_numpy_where
            ak_where = _ak_where_param(np_where)
            ak_x = ak.array(x)
            ak_out = ak.array(np_out)
            ak_res = ak_func(ak_x, ak_out, where=ak_where)

        elif use_where and name in {
            "log",
            "log2",
            "log10",
            "log1p",
            "exp",
            "expm1",
            "sign",
            "fabs",
            "sin",
            "cos",
            "tan",
            "arcsin",
            "arccos",
            "arctan",
            "sinh",
            "cosh",
            "tanh",
            "arcsinh",
            "arccosh",
            "arctanh",
        }:
            np_where = where_mask
            np_out = np.ones_like(x, dtype=ak.float64)  # these functions always output float
            np_res = np_func(x, np_out, where=np_where)  # this behavior differs from _apply_numpy_where
            ak_where = _ak_where_param(np_where)
            ak_x = ak.array(x)
            ak_out = ak.array(np_out)
            ak_res = ak_func(ak_x, ak_out, where=ak_where)

        else:
            np_res = np_func(x.astype(np.float64) if name == "fabs" else x)
            ak_res = ak_func(ak.array(x))

        # Some functions always yield float in arkouda (fabs); accept that.
        ak_np = _to_np(ak_res)
        assert isinstance(ak_np, np.ndarray)

        # Shape and value checks
        assert ak_np.shape == np_res.shape

        # isnan on non-floats in arkouda returns all False; numpy does too for ints/bools
        if np_res.dtype.kind == "b":
            np.testing.assert_array_equal(ak_np, np_res)
        else:
            np.testing.assert_allclose(ak_np, np_res, rtol=RTOL, atol=ATOL, equal_nan=True)

    np.seterr(**old_settings)  # restore original settings


def test_rad2deg_deg2rad_alignment(base_float, where_mask) -> None:
    x = base_float.copy()
    where = where_mask

    # rad2deg
    np_r2d = _apply_numpy_where(x, where, np.rad2deg)
    ak_r2d = ak.rad2deg(ak.array(x), where=_ak_where_param(where))
    _assert_array_equal_or_allclose(np_r2d, ak_r2d)

    # deg2rad
    np_d2r = _apply_numpy_where(x, where, np.deg2rad)
    ak_d2r = ak.deg2rad(ak.array(x), where=_ak_where_param(where))
    _assert_array_equal_or_allclose(np_d2r, ak_d2r)


# -------------------------
# Reductions / cumulative
# -------------------------


@pytest.mark.parametrize(
    "dtype_name,arr",
    [
        ("int64", "base_int"),
        ("uint64", "base_uint"),
        ("float64", "base_float"),
        ("bool", "base_bool"),
    ],
)
def test_cumsum_alignment(dtype_name: str, arr: str, request) -> None:
    x = request.getfixturevalue(arr)
    ak_x = ak.array(x)

    np_res = np.cumsum(x, axis=0)
    ak_res = ak.cumsum(ak_x)

    _assert_array_equal_or_allclose(np_res.astype(_to_np(ak_res).dtype), ak_res)


@pytest.mark.parametrize(
    "dtype_name,arr",
    [
        ("int64", "base_int"),
        ("uint64", "base_uint"),
        ("float64", "base_float"),
        ("bool", "base_bool"),
    ],
)
def test_cumprod_alignment(dtype_name: str, arr: str, request) -> None:
    x = request.getfixturevalue(arr)
    ak_x = ak.array(x)

    np_res = np.cumprod(x, axis=0)
    ak_res = ak.cumprod(ak_x)

    _assert_array_equal_or_allclose(np_res.astype(_to_np(ak_res).dtype), ak_res)


def test_count_nonzero_alignment(base_int, base_bool) -> None:
    assert int(ak.count_nonzero(ak.array(base_int))) == int(np.count_nonzero(base_int))
    assert int(ak.count_nonzero(ak.array(base_bool))) == int(np.count_nonzero(base_bool))


def test_median_alignment(base_int, base_float, base_bool) -> None:
    # median returns np.float64 in arkouda numeric.py
    np.testing.assert_allclose(float(ak.median(ak.array(base_int))), float(np.median(base_int)))
    np.testing.assert_allclose(
        float(ak.median(ak.array(base_float[np.isfinite(base_float)]))),
        float(np.median(base_float[np.isfinite(base_float)])),
        rtol=RTOL,
        atol=ATOL,
    )
    # bool median in arkouda sorts cast-to-int
    np.testing.assert_allclose(
        float(ak.median(ak.array(base_bool))),
        float(np.median(base_bool.astype(int))),
        rtol=RTOL,
        atol=ATOL,
    )


# -------------------------
# where / putmask
# -------------------------


def test_where_numeric_alignment(base_int, where_mask) -> None:
    cond = where_mask
    x = base_int
    y = (base_int + 7).astype(np.int64)

    np_res = np.where(cond, x, y)
    ak_res = ak.where(ak.array(cond), ak.array(x), ak.array(y))

    _assert_array_equal_or_allclose(np_res, ak_res)


def test_where_scalar_alignment(base_int, where_mask) -> None:
    cond = where_mask
    x = base_int
    scalar = 123

    np_res = np.where(cond, x, scalar)
    ak_res = ak.where(ak.array(cond), ak.array(x), scalar)

    _assert_array_equal_or_allclose(np_res, ak_res)


def test_putmask_alignment(base_int, where_mask) -> None:
    # Overwrites in place like numpy.putmask
    x = base_int.copy()
    y = (base_int * 3).astype(np.int64)
    mask = where_mask

    np_a = x.copy()
    np.putmask(np_a, mask, y)

    ak_a = ak.array(x)
    ak.putmask(ak_a, ak.array(mask), ak.array(y))

    _assert_array_equal_or_allclose(np_a, ak_a)


# -------------------------
# take / clip / min / max / array_equal
# -------------------------


def test_take_alignment_1d(base_int) -> None:
    x = base_int
    idx = np.array([0, 5, 10, 42, 199], dtype=np.int64)

    np_res = np.take(x, idx)
    ak_res = ak.take(ak.array(x), ak.array(idx))

    _assert_array_equal_or_allclose(np_res, ak_res)


@pytest.mark.skip_if_rank_not_compiled([2])
def test_take_alignment_axis_2d(rng: np.random.Generator) -> None:
    x = rng.integers(-10, 10, size=(8, 6), dtype=np.int64)
    idx = np.array([0, 2, 4], dtype=np.int64)

    np_res = np.take(x, idx, axis=1)
    ak_res = ak.take(ak.array(x), ak.array(idx), axis=1)

    _assert_array_equal_or_allclose(np_res, ak_res)


def test_clip_alignment_scalar_bounds(base_float) -> None:
    x = base_float
    lo, hi = -1.25, 2.5
    np_res = np.clip(x, lo, hi)
    ak_res = ak.clip(ak.array(x), lo, hi)
    _assert_array_equal_or_allclose(np_res, ak_res)


@pytest.mark.skip_if_rank_not_compiled([2])
def test_minimum_maximum_alignment_broadcast(base_float, rng: np.random.Generator) -> None:
    a = base_float.reshape(20, 10)
    b = rng.normal(size=(10,)).astype(np.float64)

    np_min = np.minimum(a, b)
    ak_min = ak.minimum(ak.array(a), ak.array(b))
    _assert_array_equal_or_allclose(np_min, ak_min)

    np_max = np.maximum(a, b)
    ak_max = ak.maximum(ak.array(a), ak.array(b))
    _assert_array_equal_or_allclose(np_max, ak_max)


def test_array_equal_alignment(base_float) -> None:
    x = base_float.copy()
    y = base_float.copy()

    # NaNs should make array_equal False by default
    assert ak.array_equal(ak.array(x), ak.array(y)) == np.array_equal(x, y)

    # With equal_nan True, NumPy treats corresponding NaNs as equal
    assert ak.array_equal(ak.array(x), ak.array(y), equal_nan=True) == np.array_equal(
        x, y, equal_nan=True
    )


# -------------------------
# matmul / vecdot
# -------------------------


@pytest.mark.skip_if_rank_not_compiled([2])
def test_matmul_alignment_2d(rng: np.random.Generator) -> None:
    a = rng.normal(size=(5, 3)).astype(np.float64)
    b = rng.normal(size=(3, 4)).astype(np.float64)

    np_res = np.matmul(a, b)
    ak_res = ak.matmul(ak.array(a), ak.array(b))

    _assert_array_equal_or_allclose(np_res, ak_res)


@pytest.mark.xfail(
    reason=(
        "NumPy semantics: 1D @ 1D returns a scalar. "
        "ak.matmul returns a scalar too, but is annotated as returning pdarray, "
        "triggering typeguard. This is a typing bug, not an alignment failure."
    ),
    strict=True,
)
def test_matmul_alignment_1d_1d(rng: np.random.Generator) -> None:
    a = rng.integers(-5, 5, size=20, dtype=np.int64)
    b = rng.integers(-5, 5, size=20, dtype=np.int64)

    np_res = np.matmul(a, b)  # dot
    ak_res = ak.matmul(ak.array(a), ak.array(b))

    # numpy returns scalar; arkouda returns scalar numeric
    assert int(ak_res) == int(np_res)


@pytest.mark.skip_if_rank_not_compiled([2])
def test_vecdot_alignment(rng: np.random.Generator) -> None:
    x1 = rng.normal(size=(4, 7)).astype(np.float64)
    x2 = rng.normal(size=(4, 7)).astype(np.float64)

    np_res = np.vecdot(x1, x2)  # defaults to last axis
    ak_res = ak.vecdot(ak.array(x1), ak.array(x2))

    _assert_array_equal_or_allclose(np_res, ak_res)
