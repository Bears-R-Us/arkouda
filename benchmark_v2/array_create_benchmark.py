import numpy as np
import pytest
from context import arkouda as ak

OPS = ("zeros", "ones", "randint")
TYPES = ("int64", "float64", "uint64")


def _create_ak_array(size, op, dtype, seed):
    if op == "zeros":
        a = ak.zeros(size, dtype=dtype)
    elif op == "ones":
        a = ak.ones(size, dtype=dtype)
    elif op == "randint":
        a = ak.randint(0, 2**32, size, dtype=dtype, seed=seed)

    return a


def _create_np_array(size, op, dtype, seed):
    if op == "zeros":
        a = np.zeros(size, dtype=dtype)
    elif op == "ones":
        a = np.ones(size, dtype=dtype)
    elif op == "randint":
        if seed is not None:
            np.random.seed(seed)
        if dtype == "int64":
            a = np.random.randint(1, size, size)
        elif dtype == "float64":
            a = np.random.random(size) + 0.5
        elif dtype == "uint64":
            a = np.random.randint(1, size, size, "uint64")

    return a


@pytest.mark.benchmark(group="AK Array Create")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_ak_array_create(benchmark, op, dtype):
    """
    Measures Arkouda array creation performance
    """
    cfg = ak.get_config()
    size = pytest.prob_size * cfg["numLocales"]

    if dtype in pytest.dtype:
        a = benchmark.pedantic(
            _create_ak_array, args=(size, op, dtype, pytest.seed), rounds=pytest.trials
        )

        bytes_per_sec = (a.size * a.itemsize) / benchmark.stats["mean"]
        benchmark.extra_info["Bytes per second"] = "{:.4f} GiB/sec".format(bytes_per_sec / 2**30)


@pytest.mark.benchmark(group="NP Array Create")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_np_array_create(benchmark, op, dtype):
    """
    Measures Numpy array creation performance
    """
    cfg = ak.get_config()
    size = pytest.prob_size * cfg["numLocales"]

    if pytest.numpy and dtype in pytest.dtype:
        a = benchmark.pedantic(
            _create_np_array, args=(size, op, dtype, pytest.seed), rounds=pytest.trials
        )

        bytes_per_sec = (a.size * a.itemsize) / benchmark.stats["mean"]
        benchmark.extra_info["Bytes per second"] = "{:.4f} GiB/sec".format(bytes_per_sec / 2**30)
