import numpy as np
import pytest

import arkouda as ak

from arkouda.numpy.manipulation_functions import flip, repeat, squeeze, tile


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize(
    "shape",
    [
        (0,),
        (1,),
        (5,),
        (2, 3),
        (3, 1, 4),
    ],
)
def test_flip_matches_numpy_axis_none(shape):
    # NumPy uses views when possible; Arkouda copies. Values must match.
    np_a = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    ak_a = ak.array(np_a)

    got = flip(ak_a, axis=None).to_ndarray()
    exp = np.flip(np_a, axis=None)

    assert np.array_equal(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize(
    "shape,axis",
    [
        ((5,), 0),
        ((2, 3), 0),
        ((2, 3), 1),
        ((3, 1, 4), 0),
        ((3, 1, 4), 2),
    ],
)
def test_flip_matches_numpy_single_axis(shape, axis):
    np_a = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    ak_a = ak.array(np_a)

    got = flip(ak_a, axis=axis).to_ndarray()
    exp = np.flip(np_a, axis=axis)

    assert np.array_equal(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize(
    "shape,axes",
    [
        ((2, 3), (0, 1)),
        ((3, 1, 4), (0, 2)),
    ],
)
def test_flip_matches_numpy_multi_axis(shape, axes):
    np_a = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    ak_a = ak.array(np_a)

    got = flip(ak_a, axis=axes).to_ndarray()
    exp = np.flip(np_a, axis=axes)

    assert np.array_equal(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize(
    "a,reps",
    [
        ([1, 2, 3], 0),
        ([1, 2, 3], 1),
        ([1, 2, 3], 2),
        ([1, 2, 3], [1, 2, 1]),
        ([1, 2, 3, 4], [2, 0, 1, 3]),
    ],
)
def test_repeat_matches_numpy_axis_none(a, reps):
    np_a = np.array(a, dtype=np.int64)
    ak_a = ak.array(np_a)

    got = repeat(ak_a, reps, axis=None).to_ndarray()
    exp = np.repeat(np_a, reps, axis=None)

    assert np.array_equal(got, exp)


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize(
    "shape,axis,reps",
    [
        ((2, 3), 0, 2),  # scalar reps broadcast along axis
        ((2, 3), 1, 3),
        ((2, 3), 0, [1, 2]),  # per-row repeats
        ((2, 3), 1, [1, 0, 2]),  # per-col repeats (including 0)
        ((3, 1, 4), 1, 2),  # axis with size-1
        ((3, 1, 4), 1, [0]),  # size-1 axis, list length 1 ok (constant-like)
    ],
)
def test_repeat_matches_numpy_axis_int(shape, axis, reps):
    np_a = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    ak_a = ak.array(np_a)

    got = repeat(ak_a, reps, axis=axis).to_ndarray()
    exp = np.repeat(np_a, reps, axis=axis)

    assert np.array_equal(got, exp)
    assert got.shape == exp.shape


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize(
    "shape,axis",
    [
        ((2, 3), 2),
        ((2, 3), -3),
        ((3, 1, 4), 3),
    ],
)
def test_repeat_axis_out_of_range_raises(shape, axis):
    np_a = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    ak_a = ak.array(np_a)

    with pytest.raises(IndexError):
        repeat(ak_a, 2, axis=axis)


def test_repeat_negative_repeats_raises_valueerror():
    ak_a = ak.array(np.array([1, 2, 3], dtype=np.int64))

    with pytest.raises(ValueError):
        repeat(ak_a, [-1, 1, 1], axis=None)

    # NumPy also raises for negative repeats
    with pytest.raises(ValueError):
        np.repeat(np.array([1, 2, 3]), [-1, 1, 1])


@pytest.mark.skip_if_rank_not_compiled([2])
def test_repeat_repeats_ndim_gt_1_raises_valueerror():
    ak_a = ak.array(np.array([1, 2, 3], dtype=np.int64))
    ak_reps = ak.array(np.array([[1, 2, 3]], dtype=np.int64))  # ndim=2

    with pytest.raises(ValueError):
        repeat(ak_a, ak_reps, axis=None)


@pytest.mark.skip_if_rank_not_compiled([2])
def test_repeat_repeats_size_mismatch_axis_raises_valueerror():
    np_a = np.arange(6, dtype=np.int64).reshape(2, 3)
    ak_a = ak.array(np_a)

    # axis=1 requires repeats length 3 or scalar; length 2 should fail
    with pytest.raises(ValueError):
        repeat(ak_a, [1, 2], axis=1)

    # NumPy also errors for mismatch
    with pytest.raises(ValueError):
        np.repeat(np_a, [1, 2], axis=1)


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize(
    "shape,axis",
    [
        ((1, 10, 1), None),
        ((1, 10, 1), 0),
        ((1, 10, 1), 2),
    ],
)
def test_squeeze_matches_numpy(shape, axis):
    np_a = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    ak_a = ak.array(np_a)

    got = squeeze(ak_a, axis=axis).to_ndarray()
    exp = np.squeeze(np_a, axis=axis)

    assert np.array_equal(got, exp)
    assert got.shape == exp.shape


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize(
    "shape,axis",
    [
        ((2, 3), 0),  # axis=0 is not size-1
        ((2, 1, 3), 0),
        ((2, 1, 3), 2),
        ((1, 2), 1),  # axis=1 is not size-1
    ],
)
def test_squeeze_invalid_axis_raises_like_numpy(shape, axis):
    np_a = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    ak_a = ak.array(np_a)

    # NumPy raises ValueError if a selected axis is not of size 1
    with pytest.raises(ValueError):
        np.squeeze(np_a, axis=axis)

    # Arkouda implementation should also reject (server may raise -> ValueError here)
    with pytest.raises(ValueError):
        squeeze(ak_a, axis=axis)


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize(
    "shape,reps",
    [
        ((3,), 2),
        ((3,), (2, 2)),
        ((3,), (2, 1, 2)),
        ((2, 2), 2),
        ((2, 2), (2, 1)),
        ((4,), (4, 1)),
        ((2, 3, 1), (1, 2, 3)),
    ],
)
def test_tile_matches_numpy(shape, reps):
    np_a = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    ak_a = ak.array(np_a)

    got = tile(ak_a, reps).to_ndarray()
    exp = np.tile(np_a, reps)

    assert np.array_equal(got, exp)
    assert got.shape == exp.shape


@pytest.mark.skip_if_rank_not_compiled([2])
def test_tile_accepts_list_reps_like_numpy():
    np_a = np.array([1, 2, 3], dtype=np.int64)
    ak_a = ak.array(np_a)

    got = tile(ak_a, [2, 2]).to_ndarray()
    exp = np.tile(np_a, [2, 2])

    assert np.array_equal(got, exp)
    assert got.shape == exp.shape
