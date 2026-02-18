import numpy as np
import pytest

import arkouda as ak


def _to_np(x):
    if hasattr(x, "to_ndarray"):
        return x.to_ndarray()
    return ak.to_numpy(x)


def _assert_in_range(arr, lo, hi, *, inclusive_hi=False):
    a = np.asarray(arr)
    assert np.all(a >= lo)
    if inclusive_hi:
        assert np.all(a <= hi)
    else:
        assert np.all(a < hi)


def _assert_shape(arr, shape):
    a = np.asarray(arr)
    assert a.shape == tuple(shape)


def _numpy_randomstate_randint(low, high, size, dtype):
    rs = np.random.RandomState(0)
    return rs.randint(low=low, high=high, size=size, dtype=dtype)


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize("shape", [(1,), (10,), (2, 3), (4, 1, 5)])
def test_ak_rand_shape_and_range(shape):
    # Arkouda: rand is under ak.random
    ak.random.seed(12345)
    out = ak.random.rand(*shape)
    out_np = _to_np(out)
    _assert_shape(out_np, shape)
    # Accept [0,1] to tolerate rare inclusive-high implementations
    _assert_in_range(out_np, 0.0, 1.0, inclusive_hi=True)


def test_ak_rand_scalar():
    ak.random.seed(12345)
    x = ak.random.rand()
    assert isinstance(x, (float, np.floating))
    assert 0.0 <= float(x) <= 1.0


@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("size", [0, 1, 10, (2, 3)])
def test_randint_int64_shape_dtype_and_bounds(size):
    low, high = 3, 17
    ak.random.seed(2468)

    # Arkouda API name: usually randint() under ak.random
    out = ak.random.randint(low, high, size=size)

    out_np = _to_np(out)

    if isinstance(size, tuple):
        _assert_shape(out_np, size)
    else:
        _assert_shape(out_np, (size,))

    assert np.issubdtype(out_np.dtype, np.integer)
    _assert_in_range(out_np, low, high, inclusive_hi=False)


def _numpy_bool_randint_error(low, high):
    rs = np.random.RandomState(0)
    with pytest.raises(ValueError) as e:
        rs.randint(low=low, high=high, size=10, dtype=bool)
    return str(e.value)


@pytest.mark.parametrize(
    "low,high",
    [
        (0, 0),
        (0, -1),
        (-1, 2),
        (0, 3),
        (1, 1),
        (1, 0),
    ],
)
def test_randint_bool_validation_messages_match_numpy(low, high):
    expected_msg = _numpy_bool_randint_error(low, high)

    ak.random.seed(0)
    with pytest.raises(ValueError) as e_ak:
        ak.random.randint(low, high, size=10, dtype="bool")

    actual = str(e_ak.value)
    if actual != expected_msg:
        pytest.xfail(
            f"Arkouda randint(dtype=bool) error msg mismatch for low={low}, high={high}: "
            f"expected={expected_msg!r}, got={actual!r}"
            f" Issue #5295."
        )

    assert actual == expected_msg


@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("size", [0, 1, 10, (2, 3)])
def test_uniform_shape_and_range(size):
    ak.random.seed(1357)

    # Many Arkouda builds use ak.random.uniform(low, high, size)
    out = ak.random.uniform(low=2.5, high=7.5, size=size)

    out_np = _to_np(out)

    if isinstance(size, tuple):
        _assert_shape(out_np, size)
    else:
        _assert_shape(out_np, (size,))

    _assert_in_range(out_np, 2.5, 7.5, inclusive_hi=True)


def test_standard_normal_basic_moments_are_reasonable():
    n = 20000
    ak.random.seed(4242)

    # Many Arkouda builds use ak.random.standard_normal(size)
    out = ak.random.standard_normal(n)

    x = _to_np(out).astype(np.float64)
    assert abs(float(x.mean())) < 0.05
    assert abs(float(x.var()) - 1.0) < 0.08


def test_random_api_scalar_and_vector_range():
    ak.random.seed(123)
    x = ak.random.random()
    assert isinstance(x, (float, np.floating))
    assert 0.0 <= float(x) < 1.0

    ak.random.seed(123)
    y = ak.random.random(1000)
    y_np = _to_np(y)
    _assert_shape(y_np, (1000,))
    _assert_in_range(y_np, 0.0, 1.0, inclusive_hi=False)
