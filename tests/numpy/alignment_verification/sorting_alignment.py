from __future__ import annotations

import numpy as np
import pytest

import arkouda as ak


DTYPES = [np.int64, np.uint64, np.float64]
SHAPES = [
    (0,),
    (1,),
    (10,),
    (3, 4),
    (2, 3, 4),
]


def _make_np_array(dtype, shape, seed: int):
    rng = np.random.default_rng(seed)
    if dtype == np.float64:
        # Avoid NaNs/Infs for clean alignment expectations
        x = rng.normal(size=shape)
        return x
    if dtype == np.int64:
        return rng.integers(-50, 50, size=shape, dtype=np.int64)
    if dtype == np.uint64:
        return rng.integers(0, 100, size=shape, dtype=np.uint64)
    raise AssertionError(f"Unhandled dtype {dtype}")


def _np_argsort_desc(a: np.ndarray, axis: int):
    # Descending indices are just ascending indices flipped along the axis
    perm = np.argsort(a, axis=axis, kind="stable")
    return np.flip(perm, axis=axis)


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("axis", [0, -1])
def test_sort_matches_numpy(dtype, shape, axis):
    a_np = _make_np_array(dtype, shape, seed=123)
    a_ak = ak.array(a_np)

    got = ak.sort(a_ak, axis=axis).to_ndarray()
    exp = np.sort(a_np, axis=axis)

    assert got.dtype == exp.dtype
    assert np.array_equal(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("ascending", [True, False])
def test_argsort_matches_numpy(dtype, shape, axis, ascending):
    a_np = _make_np_array(dtype, shape, seed=321)
    a_ak = ak.array(a_np)

    got = ak.argsort(a_ak, axis=axis, ascending=ascending).to_ndarray()

    if ascending:
        exp = np.argsort(a_np, axis=axis, kind="stable")
    else:
        exp = _np_argsort_desc(a_np, axis=axis)

    assert got.dtype == exp.dtype
    assert np.array_equal(got, exp)

    # Also validate that applying the permutation actually sorts
    # (NumPy-based expectation)
    got_sorted = np.take_along_axis(a_np, got, axis=axis)
    exp_sorted = np.take_along_axis(a_np, exp, axis=axis)
    assert np.array_equal(got_sorted, exp_sorted)


@pytest.mark.parametrize("dtype", DTYPES)
def test_coargsort_matches_numpy_lexsort(dtype):
    # coargsort: primary key is arrays[0], secondary arrays[1], ...
    # NumPy lexsort uses the *last* key as primary, so we pass reversed order.
    n = 50
    a0 = _make_np_array(dtype, (n,), seed=1)
    a1 = _make_np_array(dtype, (n,), seed=2)
    a2 = _make_np_array(dtype, (n,), seed=3)

    ak0, ak1, ak2 = ak.array(a0), ak.array(a1), ak.array(a2)

    got_asc = ak.coargsort([ak0, ak1, ak2], ascending=True).to_ndarray()
    exp_asc = np.lexsort((a2, a1, a0))  # reverse order for lexsort

    assert np.array_equal(got_asc, exp_asc)

    # Arkouda flips the permutation for descending when max_dim == 1
    got_desc = ak.coargsort([ak0, ak1, ak2], ascending=False).to_ndarray()
    exp_desc = exp_asc[::-1]
    assert np.array_equal(got_desc, exp_desc)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted_matches_numpy(dtype, side):
    # a must be 1D sorted ascending
    a_np = _make_np_array(dtype, (100,), seed=999)
    a_np.sort()
    a_ak = ak.array(a_np)

    # scalar v
    if dtype == np.float64:
        v_scalar = float(a_np[50])
    elif dtype == np.int64:
        v_scalar = int(a_np[50])
    else:
        # Keep as numpy scalar so arkouda preserves dtype on normalization
        v_scalar = np.uint64(a_np[50])

    got_scalar = ak.searchsorted(a_ak, v_scalar, side=side)
    exp_scalar = int(np.searchsorted(a_np, v_scalar, side=side))
    assert got_scalar == exp_scalar

    # vector v (unsorted)
    v_np = _make_np_array(dtype, (40,), seed=1001)
    v_ak = ak.array(v_np)

    got_vec = ak.searchsorted(a_ak, v_ak, side=side, x2_sorted=False).to_ndarray()
    exp_vec = np.searchsorted(a_np, v_np, side=side)
    assert np.array_equal(got_vec, exp_vec)

    # vector v (sorted) should match regardless of x2_sorted flag
    v_np_sorted = np.sort(v_np)
    v_ak_sorted = ak.array(v_np_sorted)

    got_sorted_false = ak.searchsorted(a_ak, v_ak_sorted, side=side, x2_sorted=False).to_ndarray()
    got_sorted_true = ak.searchsorted(a_ak, v_ak_sorted, side=side, x2_sorted=True).to_ndarray()
    exp_sorted = np.searchsorted(a_np, v_np_sorted, side=side)

    assert np.array_equal(got_sorted_false, exp_sorted)
    assert np.array_equal(got_sorted_true, exp_sorted)


@pytest.mark.skip_if_rank_not_compiled([2])
def test_searchsorted_rejects_non_1d():
    a_ak = ak.arange(12).reshape((3, 4))
    with pytest.raises(ValueError):
        ak.searchsorted(a_ak, 3)


@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("dtype", DTYPES)
def test_sort_and_argsort_invalid_axis(dtype):
    a_np = _make_np_array(dtype, (3, 4), seed=7)
    a_ak = ak.array(a_np)

    with pytest.raises(IndexError):
        ak.sort(a_ak, axis=2)
    with pytest.raises(IndexError):
        ak.argsort(a_ak, axis=2)
