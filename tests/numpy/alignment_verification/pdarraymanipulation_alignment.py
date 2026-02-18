import numpy as np
import pytest

import arkouda as ak

from arkouda.numpy.pdarrayclass import pdarray
from arkouda.numpy.pdarraymanipulation import append, delete, hstack, vstack


# -----------------------------
# Helpers
# -----------------------------
def _ak_to_np(x):
    """
    Convert Arkouda pdarray (possibly nested / multi-d) to a NumPy array.
    We prefer to_ndarray() when available; fall back to to_list().
    """
    # Many Arkouda objects support to_ndarray(); pdarray does.
    if hasattr(x, "to_ndarray"):
        return x.to_ndarray()
    return np.array(x.to_list())


def _assert_np_equal(got, exp):
    got_np = _ak_to_np(got) if isinstance(got, pdarray) else np.asarray(got)
    exp_np = np.asarray(exp)

    assert got_np.shape == exp_np.shape
    assert got_np.dtype == exp_np.dtype

    # Handle NaNs in a NumPy-version-stable way
    if np.issubdtype(exp_np.dtype, np.floating) or np.issubdtype(exp_np.dtype, np.complexfloating):
        np.testing.assert_allclose(got_np, exp_np, rtol=0, atol=0, equal_nan=True)
    else:
        np.testing.assert_array_equal(got_np, exp_np)


def _mk_cases_1d_same_len():
    return [
        (np.array([1, 2, 3], dtype=np.int64), np.array([4, 5, 6], dtype=np.int64)),
        (np.array([1, 2, 3], dtype=np.int64), np.array([4.5, 5.5, 6.5], dtype=np.float64)),
        (np.array([np.nan, 1.0], dtype=np.float64), np.array([2.0, np.nan], dtype=np.float64)),
        (np.array([True, False], dtype=bool), np.array([False, True], dtype=bool)),
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
    ]


def _mk_cases_2d():
    return [
        (np.array([[1], [2], [3]], dtype=np.int64), np.array([[4], [5], [6]], dtype=np.int64)),
        (np.array([[1, 2]], dtype=np.int64), np.array([[3, 4]], dtype=np.int64)),
        (np.array([[1, 2]], dtype=np.int64), np.array([[3.0, 4.0]], dtype=np.float64)),
        (np.array([[np.nan, 1.0]], dtype=np.float64), np.array([[2.0, np.nan]], dtype=np.float64)),
    ]


def _to_ak(x: np.ndarray):
    # ak.array handles numpy arrays; for multi-d it yields Arkouda "multi-d" pdarray-like.
    return ak.array(x)


# -----------------------------
# hstack alignment
# -----------------------------
@pytest.mark.parametrize("a,b", _mk_cases_1d_same_len())
def test_hstack_1d_alignment(a, b):
    ak_a, ak_b = _to_ak(a), _to_ak(b)

    got = hstack((ak_a, ak_b))
    exp = np.hstack((a, b))

    _assert_np_equal(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("a,b", _mk_cases_2d())
def test_hstack_2d_alignment(a, b):
    ak_a, ak_b = _to_ak(a), _to_ak(b)

    got = hstack((ak_a, ak_b))
    exp = np.hstack((a, b))

    _assert_np_equal(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2])
def test_hstack_dim_mismatch_raises():
    a = _to_ak(np.array([1, 2, 3], dtype=np.int64))
    b = _to_ak(np.array([[4], [5], [6]], dtype=np.int64))
    with pytest.raises(ValueError, match="same number of dimensions"):
        hstack((a, b))


def test_hstack_casting_not_supported():
    a = _to_ak(np.array([1, 2, 3], dtype=np.int64))
    b = _to_ak(np.array([4, 5], dtype=np.int64))
    with pytest.raises(NotImplementedError):
        hstack((a, b), casting="unsafe")


# -----------------------------
# vstack alignment
# -----------------------------
@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("a,b", _mk_cases_1d_same_len())
def test_vstack_1d_alignment(a, b):
    ak_a, ak_b = _to_ak(a), _to_ak(b)
    got = vstack((ak_a, ak_b))
    exp = np.vstack((a, b))
    _assert_np_equal(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("a,b", _mk_cases_2d())
def test_vstack_2d_alignment(a, b):
    ak_a, ak_b = _to_ak(a), _to_ak(b)

    got = vstack((ak_a, ak_b))
    exp = np.vstack((a, b))

    _assert_np_equal(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2])
def test_vstack_dim_mismatch_raises():
    a_np = np.array([1, 2, 3], dtype=np.int64)
    b_np = np.array([[4], [5], [6]], dtype=np.int64)

    a = _to_ak(a_np)
    b = _to_ak(b_np)

    # NumPy: must raise for mismatched dimensions
    with pytest.raises(ValueError):
        np.vstack((a_np, b_np))

    # Arkouda: currently raises RuntimeError from server; message is shape-related
    with pytest.raises((ValueError, RuntimeError), match="same shape|shape except|concatenation axis"):
        vstack((a, b))


def test_vstack_casting_not_supported():
    a = _to_ak(np.array([1, 2, 3], dtype=np.int64))
    b = _to_ak(np.array([4, 5], dtype=np.int64))
    with pytest.raises(NotImplementedError):
        vstack((a, b), casting="unsafe")


# -----------------------------
# delete alignment
# -----------------------------
@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize(
    "arr,obj,axis",
    [
        # 1D basics
        (np.array([1, 2, 3, 4], dtype=np.int64), 0, None),
        (np.array([1, 2, 3, 4], dtype=np.int64), -1, None),
        (np.array([1, 2, 3, 4], dtype=np.int64), slice(0, 4, 2), None),
        (np.array([1, 2, 3, 4], dtype=np.int64), [1, 3], None),
        (np.array([1, 2, 3, 4], dtype=np.int64), np.array([True, False, True, False]), None),
        # 2D axis cases
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64), 0, 0),
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64), 1, 1),
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64), slice(0, 3, 2), 1),
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64), [0, 1], 0),
    ],
)
def test_delete_alignment(arr, obj, axis):
    ak_arr = _to_ak(arr)

    # convert obj to ak where relevant
    if isinstance(obj, np.ndarray) and obj.dtype == bool:
        ak_obj = ak.array(obj.tolist())
    elif isinstance(obj, (list, tuple)):
        ak_obj = obj  # delete() accepts Sequence[int]/Sequence[bool]
    else:
        ak_obj = obj  # int or slice

    got = delete(ak_arr, ak_obj, axis=axis)
    exp = np.delete(arr, obj, axis=axis)

    _assert_np_equal(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2])
def test_delete_axis_none_flattens_like_numpy():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    ak_arr = _to_ak(arr)

    got = delete(ak_arr, [1, 3], axis=None)
    exp = np.delete(arr, [1, 3], axis=None)

    _assert_np_equal(got, exp)


# -----------------------------
# append alignment
# -----------------------------
@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize(
    "arr,values,axis",
    [
        # axis=None -> flatten both (NumPy behavior)
        (np.array([1, 2, 3], dtype=np.int64), np.array([[4, 5], [6, 7]], dtype=np.int64), None),
        (np.array([[1, 2], [3, 4]], dtype=np.int64), np.array([5, 6], dtype=np.int64), None),
        # axis specified -> shapes must align except on axis
        (np.array([[1, 2], [3, 4]], dtype=np.int64), np.array([[5, 6]], dtype=np.int64), 0),
        (np.array([[1, 2], [3, 4]], dtype=np.int64), np.array([[5], [6]], dtype=np.int64), 1),
        # dtype promotion
        (np.array([1, 2, 3], dtype=np.int64), np.array([4.5], dtype=np.float64), None),
    ],
)
def test_append_alignment(arr, values, axis):
    ak_arr = _to_ak(arr)
    ak_values = _to_ak(values)

    got = append(ak_arr, ak_values, axis=axis)
    exp = np.append(arr, values, axis=axis)

    _assert_np_equal(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2])
def test_append_axis_dim_mismatch_raises():
    arr = _to_ak(np.array([1, 2, 3], dtype=np.int64))
    values = _to_ak(np.array([[4], [5]], dtype=np.int64))
    with pytest.raises(ValueError, match="same number of dimensions"):
        append(arr, values, axis=0)


@pytest.mark.skip_if_rank_not_compiled([2])
def test_append_axis_out_of_bounds_raises():
    arr = _to_ak(np.array([[1, 2], [3, 4]], dtype=np.int64))
    values = _to_ak(np.array([[5, 6]], dtype=np.int64))
    with pytest.raises(ValueError, match="out of bounds"):
        append(arr, values, axis=5)
