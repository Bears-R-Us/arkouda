import math

import numpy as np
import pytest

import arkouda as ak


SEED = 12345


def assert_scalar_close(a, b, *, rtol=0.0, atol=0.0):
    # Handle ints/bools exactly; floats with tolerance
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        assert bool(a) == bool(b)
        return
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        assert int(a) == int(b)
        return
    # float-ish
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert float(a) == pytest.approx(float(b), rel=rtol, abs=atol)


@pytest.mark.parametrize(
    "method_name, kwargs",
    [
        ("random", {}),
        ("uniform", {"low": -2.0, "high": 3.0}),
        ("standard_normal", {}),
        ("standard_exponential", {"method": "zig"}),
        ("standard_exponential", {"method": "inv"}),
        ("integers", {"low": 0, "high": 10, "endpoint": False}),
        ("integers", {"low": 5, "high": None, "endpoint": False}),  # NumPy interprets as [0,5)
        ("logistic", {"loc": 0.1, "scale": 2.0}),
        ("standard_gamma", {"shape": 2.5}),
        ("poisson", {"lam": 3}),
    ],
)
def test_scalar_delegation_matches_numpy(method_name, kwargs):
    """
    When size is None, Arkouda delegates many RNG calls to NumPy.
    Those scalar results should match NumPy exactly for the same seed
    *if the RNG stream is aligned*.
    """
    rng_np = np.random.default_rng(SEED)
    rng_ak = ak.random.default_rng(SEED)

    f_np = getattr(rng_np, method_name)
    f_ak = getattr(rng_ak, method_name)

    got_np = f_np(**kwargs)
    got_ak = f_ak(**kwargs)

    # Exact for ints/bools; very tight for floats
    assert_scalar_close(got_ak, got_np, rtol=0.0, atol=0.0)


@pytest.mark.skip_if_rank_not_compiled([2, 3])
@pytest.mark.parametrize(
    "size",
    [1, 10, (2, 3), (3, 2, 2)],
)
def test_standard_normal_shape_and_dtype(size):
    rng = ak.random.default_rng(SEED)

    # Arkouda limitation: multidimensional arrays only support BOX method
    method = "box" if isinstance(size, tuple) else "zig"

    out = rng.standard_normal(size=size, method=method)
    assert isinstance(out, ak.pdarray)
    assert out.dtype == ak.float64
    assert out.shape == (size,) if isinstance(size, int) else size


@pytest.mark.skip_if_rank_not_compiled([2])
@pytest.mark.parametrize("size", [1, 10, (2, 3)])
def test_uniform_bounds(size):
    rng = ak.random.default_rng(SEED)
    low, high = -1.5, 2.25
    out = rng.uniform(low=low, high=high, size=size)
    assert isinstance(out, ak.pdarray)
    assert out.dtype == ak.float64
    assert (out >= low).all()
    assert (out < high).all()


@pytest.mark.parametrize(
    "dtype",
    [ak.int64, ak.uint64],
)
def test_integers_dtype_and_bounds(dtype):
    rng = ak.random.default_rng(SEED)
    out = rng.integers(10, 20, size=1000, dtype=dtype)
    assert isinstance(out, ak.pdarray)
    assert out.dtype == dtype
    assert (out >= 10).all()
    assert (out <= 19).all()


@pytest.mark.xfail(
    reason="Bug: integers dtype guard uses `is` after dtype normalization; "
    "float dtypes not rejected. Issue #5298."
)
def test_integers_rejects_float64_dtype():
    rng = ak.random.default_rng(SEED)
    with pytest.raises(TypeError):
        rng.integers(0, 10, size=10, dtype=ak.float64)


@pytest.mark.parametrize("bad_size", [-1, (2, -3)])
def test_standard_normal_rejects_negative_size(bad_size):
    rng = ak.random.default_rng(SEED)
    with pytest.raises(ValueError):
        rng.standard_normal(size=bad_size)


def test_exponential_rejects_negative_scale_scalar():
    rng = ak.random.default_rng(SEED)
    with pytest.raises(TypeError):
        rng.exponential(scale=-0.1, size=10)


def test_poisson_size_zero_returns_empty_int64():
    rng = ak.random.default_rng(SEED)
    out = rng.poisson(lam=3.0, size=0)
    assert isinstance(out, ak.pdarray)
    assert out.dtype == ak.int64
    assert out.size == 0


def test_reproducible_arrays_same_seed_same_result():
    # Reproducibility check within Arkouda, independent of NumPy
    rng1 = ak.random.default_rng(SEED)
    rng2 = ak.random.default_rng(SEED)

    a1 = rng1.uniform(size=1000)
    a2 = rng2.uniform(size=1000)
    assert (a1 == a2).all()


@pytest.mark.xfail(
    reason="Mixed scalar (NumPy-delegated) and server RNG calls may not share a single stream yet."
)
def test_mixed_call_sequence_matches_numpy_stream():
    """
    This is the 'gotcha' alignment test: NumPy advances its stream for array draws.
    If Arkouda draws arrays on the server but scalars via local NumPy without syncing,
    the scalar sequence after an array draw will diverge from NumPy.
    """
    rng_np = np.random.default_rng(SEED)
    rng_ak = ak.random.default_rng(SEED)

    # both do an array draw (NumPy consumes stream)
    _ = rng_np.uniform(size=10)
    _ = rng_ak.uniform(size=10)

    # next scalar should match if the streams are truly aligned
    got_np = rng_np.random()
    got_ak = rng_ak.random()
    assert_scalar_close(got_ak, got_np, rtol=0.0, atol=0.0)
