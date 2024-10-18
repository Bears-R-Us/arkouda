import arkouda as ak
import pytest

DTYPES = ["int64", "float64", "bigint"]


def run_test(a, b, alpha):
    return a + b * alpha

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="ak_stream")
@pytest.mark.parametrize("dtype", DTYPES)
def bench_ak_stream(benchmark, dtype):
    """
    Measures performance of stream using "int64", "float64", and "bigint" types.

    """
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]
    # default tot_bytes to ones case
    nBytes = N * 8 * 3

    if dtype in ["int64", "float64"]:
        if pytest.random or pytest.seed is not None:
            if dtype == "int64":
                a = ak.randint(0, 2 ** 32, N, seed=pytest.seed)
                b = ak.randint(0, 2 ** 32, N, seed=pytest.seed)
                benchmark.pedantic(run_test, args=(a, b, pytest.alpha), rounds=pytest.trials)
            elif dtype == "float64":
                a = ak.randint(0, 1, N, dtype=ak.float64, seed=pytest.seed)
                b = ak.randint(0, 1, N, dtype=ak.float64, seed=pytest.seed)
                benchmark.pedantic(run_test, args=(a, b, pytest.alpha), rounds=pytest.trials)
        else:
            a = ak.ones(N, dtype=dtype)
            b = ak.ones(N, dtype=dtype)
            benchmark.pedantic(run_test, args=(a, b, pytest.alpha), rounds=pytest.trials)

    else:  # bigint
        if pytest.random or pytest.seed is not None:
            a1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
            a2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
            a = ak.bigint_from_uint_arrays([a1, a2], max_bits=pytest.max_bits)
            b1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
            b2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
            b = ak.bigint_from_uint_arrays([b1, b2], max_bits=pytest.max_bits)
            # update tot_bytes to account for using 2 uint64
            nBytes *= 2
        else:
            a = ak.bigint_from_uint_arrays([ak.ones(N, dtype=ak.uint64)], max_bits=pytest.max_bits)
            b = ak.bigint_from_uint_arrays([ak.ones(N, dtype=ak.uint64)], max_bits=pytest.max_bits)

        benchmark.pedantic(run_test, args=(a, b, int(pytest.alpha)), rounds=pytest.trials)

    benchmark.extra_info["description"] = f"Measures performance of stream using {dtype} types."
    benchmark.extra_info["problem_size"] = pytest.prob_size
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nBytes / benchmark.stats["mean"]) / 2 ** 30
    )