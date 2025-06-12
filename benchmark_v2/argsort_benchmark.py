import numpy as np
import pytest

import arkouda as ak

TYPES = ("int64", "uint64", "float64", "str")


@pytest.mark.benchmark(group="arkouda_argsort")
@pytest.mark.parametrize("dtype", TYPES)
def bench_argsort(benchmark, dtype):
    """
    Measure ArgSort performance. Runs for each dtype in TYPES

    Note
    -----
    str dtype is significantly slower than numerics
    """
    cfg = ak.get_config()
    N = 10**4 if pytest.correctness_only else pytest.prob_size * cfg["numLocales"]

    if dtype in pytest.dtype:
        if dtype == "int64":
            a = ak.randint(0, 2**32, N, seed=pytest.seed)
        elif dtype == "uint64":
            a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
        elif dtype == "float64":
            a = ak.randint(0, 1, N, dtype=ak.float64, seed=pytest.seed)
        elif dtype == "str":
            a = ak.random_strings_uniform(1, 16, N, seed=pytest.seed)

        if dtype == "str":
            nbytes = a.nbytes * a.entry.itemsize
        else:
            nbytes = a.size * a.itemsize

        if pytest.numpy:
            a = a.to_ndarray()
            result = benchmark.pedantic(np.argsort, args=[a], rounds=pytest.trials)
        else:
            result = benchmark.pedantic(ak.argsort, args=[a], rounds=pytest.trials)

        if pytest.correctness_only and not pytest.numpy and dtype != "str":
            a_np = a.to_ndarray()
            expected = np.argsort(a_np)
            np.testing.assert_array_equal(result.to_ndarray(), expected)

        benchmark.extra_info["description"] = "Measures the performance of argsort"
        benchmark.extra_info["problem_size"] = N
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )
