import numpy as np
import pytest

import arkouda as ak

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


@pytest.mark.benchmark(group="Array_Create")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_array_create(benchmark, op, dtype):
    """
    Measures array creation performance (Arkouda or NumPy based on flags)
    """
    N = pytest.N

    if dtype in pytest.dtype:
        if pytest.numpy:

            def create_array():
                a = _create_np_array(N, op, dtype, pytest.seed)
                return a.size * a.itemsize
        else:

            def create_array():
                a = _create_ak_array(N, op, dtype, pytest.seed)
                return a.size * a.itemsize

        nbytes = benchmark.pedantic(create_array, rounds=pytest.trials)

        benchmark.extra_info["description"] = (
            f"Measures performance of {'NumPy' if pytest.numpy else 'Arkouda'} array creation"
        )
        benchmark.extra_info["problem_size"] = N
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )
