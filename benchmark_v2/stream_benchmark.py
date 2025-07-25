import numpy as np
import pytest

import arkouda as ak

DTYPES = ["int64", "float64", "bigint"]


def run_test(a, b, alpha):
    return a + b * alpha


def check_numpy_equivalence(a, b, alpha, result):
    a_np, b_np = a.to_ndarray(), b.to_ndarray()
    expected = a_np + b_np * alpha
    np.testing.assert_allclose(result.to_ndarray(), expected, rtol=1e-10)


@pytest.mark.benchmark(group="stream")
@pytest.mark.parametrize("dtype", ["int64", "float64"])
def bench_stream(benchmark, dtype):
    N = pytest.N
    nBytes = N * 8 * 3

    if pytest.random or pytest.seed is not None:
        if dtype == "int64":
            a = ak.randint(0, 2**32, N, seed=pytest.seed)
            b = ak.randint(0, 2**32, N, seed=pytest.seed)
        elif dtype == "float64":
            a = ak.randint(0, 1, N, dtype=ak.float64, seed=pytest.seed)
            b = ak.randint(0, 1, N, dtype=ak.float64, seed=pytest.seed)
    else:
        a = ak.ones(N, dtype=dtype)
        b = ak.ones(N, dtype=dtype)

    if pytest.numpy:
        a, b = a.to_ndarray(), b.to_ndarray()

    benchmark.pedantic(run_test, args=(a, b, pytest.alpha), rounds=pytest.trials)

    benchmark.extra_info["description"] = f"Measures performance of stream using {dtype} types."
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nBytes / benchmark.stats["mean"]) / 2**30
    )


@pytest.mark.benchmark(group="stream")
@pytest.mark.parametrize("dtype", ["bigint"])
def bench_bigint_stream(benchmark, dtype):
    N = pytest.N
    nBytes = N * 8 * 3

    if pytest.random or pytest.seed is not None:
        a1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
        a2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
        a = ak.bigint_from_uint_arrays([a1, a2], max_bits=pytest.max_bits)

        b1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
        b2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
        b = ak.bigint_from_uint_arrays([b1, b2], max_bits=pytest.max_bits)

        nBytes *= 2
    else:
        a = ak.bigint_from_uint_arrays([ak.ones(N, dtype=ak.uint64)], max_bits=pytest.max_bits)
        b = ak.bigint_from_uint_arrays([ak.ones(N, dtype=ak.uint64)], max_bits=pytest.max_bits)

    benchmark.pedantic(run_test, args=(a, b, int(pytest.alpha)), rounds=pytest.trials)

    # Can't do numpy comparison for bigint yet
    benchmark.extra_info["description"] = f"Measures performance of stream using {dtype} types."
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nBytes / benchmark.stats["mean"]) / 2**30
    )
