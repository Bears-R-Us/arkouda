from __future__ import annotations

import numpy as np
import pytest

import arkouda as ak


# NOTE:
# - These tests focus only on functions with a clear NumPy equivalent.
# - Anything Arkouda-specific (registration, server buffer identity, etc.) is skipped by design.


def _to_np(x):
    """Convert Arkouda outputs to NumPy arrays for comparison."""
    # pdarray / ArrayView-like objects
    if hasattr(x, "to_ndarray"):
        return x.to_ndarray()

    # some Arkouda objects expose to_list
    if hasattr(x, "to_list"):
        return np.array(x.to_list())

    # scalars / numpy / python containers
    return np.asarray(x)


# -----------------------------------------------------------------------------
# shape (ak.shape)  <->  numpy.shape
# -----------------------------------------------------------------------------
@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize(
    "obj",
    [
        0,
        3.14,
        [1, 2, 3],
        [[1, 2], [3, 4]],
        np.array([1, 2, 3]),
        np.array([[1, 2], [3, 4]]),
    ],
)
def test_shape_matches_numpy(obj):
    # numpy
    expected = np.shape(obj)

    # arkouda: shape accepts iterables/scalars and will materialize iterables via ak.array
    got = ak.shape(obj)

    assert got == expected


@pytest.mark.skip_if_rank_not_compiled([2])
def test_shape_matches_numpy_for_ak_arrays():
    a = ak.arange(12).reshape(3, 4)
    assert ak.shape(a) == np.shape(np.arange(12).reshape(3, 4))


# -----------------------------------------------------------------------------
# broadcast_shapes (ak.broadcast_shapes)  <->  numpy.broadcast_shapes
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shapes",
    [
        ((3,), (1,)),
        ((1, 3), (3,)),
        ((1, 2, 3), (4, 1, 3), (4, 2, 1)),
        ((), (5,)),
        ((2, 1), (1, 3)),
    ],
)
def test_broadcast_shapes_matches_numpy(shapes):
    expected = np.broadcast_shapes(*shapes)
    got = ak.broadcast_shapes(*shapes)
    assert got == expected


@pytest.mark.parametrize(
    "shapes",
    [
        ((2,), (3,)),
        ((2, 2), (3, 2)),
        ((2, 3), (3, 2)),
    ],
)
def test_broadcast_shapes_incompatible_raises_like_numpy(shapes):
    with pytest.raises(ValueError):
        np.broadcast_shapes(*shapes)
    with pytest.raises(ValueError):
        ak.broadcast_shapes(*shapes)


# -----------------------------------------------------------------------------
# broadcast_dims (ak.broadcast_dims)  <->  numpy.broadcast_shapes (2-arg case)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sa,sb",
    [
        ((5, 1), (1, 3)),
        ((4,), (3, 1)),
        ((), (7,)),
        ((2, 1, 3), (1, 4, 1)),
    ],
)
def test_broadcast_dims_matches_numpy(sa, sb):
    expected = np.broadcast_shapes(sa, sb)
    got = ak.broadcast_dims(sa, sb)
    assert got == expected


@pytest.mark.parametrize(
    "sa,sb",
    [
        ((2,), (3,)),
        ((2, 2), (3, 2)),
    ],
)
def test_broadcast_dims_incompatible_raises(sa, sb):
    with pytest.raises(ValueError):
        np.broadcast_shapes(sa, sb)
    with pytest.raises(ValueError):
        ak.broadcast_dims(sa, sb)


# -----------------------------------------------------------------------------
# broadcast_to (ak.broadcast_to)  <->  numpy.broadcast_to
# -----------------------------------------------------------------------------
@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize(
    "value,shape",
    [
        (5, 3),
        (5, (2, 3)),
        (2.5, 4),
        (2.5, (2, 2)),
    ],
)
def test_broadcast_to_scalar_matches_numpy(value, shape):
    expected = np.broadcast_to(value, shape)
    got = ak.broadcast_to(value, shape)
    assert np.array_equal(_to_np(got), np.array(expected))


@pytest.mark.skip_if_rank_not_compiled([2])
def test_broadcast_to_pdarray_matches_numpy_tuple_shape():
    x_np = np.arange(5)
    x_ak = ak.arange(5)

    expected = np.broadcast_to(x_np, (2, 5))
    got = ak.broadcast_to(x_ak, (2, 5))

    assert np.array_equal(_to_np(got), np.array(expected))


def test_broadcast_to_pdarray_int_shape_passthrough_matches_numpy():
    # Arkouda special-cases: if x is 1D and size matches, return x unchanged
    x_np = np.arange(6)
    x_ak = ak.arange(6)

    expected = np.broadcast_to(x_np, (6,))
    got = ak.broadcast_to(x_ak, 6)

    assert np.array_equal(_to_np(got), np.array(expected))


def test_broadcast_to_pdarray_int_shape_mismatch_raises():
    x_ak = ak.arange(6)
    with pytest.raises(ValueError):
        ak.broadcast_to(x_ak, 7)


# -----------------------------------------------------------------------------
# broadcast_arrays (ak.broadcast_arrays)  <->  numpy.broadcast_arrays
# -----------------------------------------------------------------------------
@pytest.mark.skip_if_rank_not_compiled([3])
def test_broadcast_arrays_matches_numpy():
    a_np = np.arange(10).reshape(1, 2, 5)
    b_np = np.arange(20).reshape(4, 1, 5)

    a_ak = ak.arange(10).reshape(1, 2, 5)
    b_ak = ak.arange(20).reshape(4, 1, 5)

    exp_a, exp_b = np.broadcast_arrays(a_np, b_np)
    got_a, got_b = ak.broadcast_arrays(a_ak, b_ak)

    assert np.array_equal(_to_np(got_a), np.array(exp_a))
    assert np.array_equal(_to_np(got_b), np.array(exp_b))


@pytest.mark.skip_if_rank_not_compiled([2])
def test_broadcast_arrays_incompatible_raises_like_numpy():
    a_np = np.zeros((2, 3))
    b_np = np.zeros((3, 2))

    a_ak = ak.zeros((2, 3))
    b_ak = ak.zeros((3, 2))

    with pytest.raises(ValueError):
        np.broadcast_arrays(a_np, b_np)
    with pytest.raises(ValueError):
        ak.broadcast_arrays(a_ak, b_ak)


# -----------------------------------------------------------------------------
# invert_permutation (ak.invert_permutation)  <->  NumPy inverse-permutation idiom
# -----------------------------------------------------------------------------
def _np_invert_permutation(perm: np.ndarray) -> np.ndarray:
    # For a valid permutation of [0..N-1], the inverse satisfies inv[perm[i]] = i
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size, dtype=perm.dtype)
    return inv


@pytest.mark.parametrize(
    "perm_list",
    [
        [0],
        [1, 0],
        [2, 0, 3, 1],
        [3, 2, 1, 0],
        [0, 2, 1, 3, 4],
    ],
)
def test_invert_permutation_matches_numpy(perm_list):
    perm_np = np.array(perm_list, dtype=np.int64)
    perm_ak = ak.array(perm_list)

    expected = _np_invert_permutation(perm_np)
    got = ak.invert_permutation(perm_ak)

    assert np.array_equal(_to_np(got), expected)
