from benchmark_utils import calc_num_bytes
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
    N = pytest.N

    if dtype in pytest.dtype:
        if dtype == "int64":
            a = ak.randint(0, 2**32, N, seed=pytest.seed)
        elif dtype == "uint64":
            a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
        elif dtype == "float64":
            a = ak.randint(0, 1, N, dtype=ak.float64, seed=pytest.seed)
        elif dtype == "str":
            a = ak.random_strings_uniform(1, 16, N, seed=pytest.seed)

        num_bytes = calc_num_bytes(a)

        if pytest.numpy:
            a = a.to_ndarray()
            benchmark.pedantic(np.argsort, args=[a], rounds=pytest.trials)
        else:
            benchmark.pedantic(ak.argsort, args=[a], rounds=pytest.trials)

        benchmark.extra_info["description"] = "Measures the performance of argsort"
        benchmark.extra_info["problem_size"] = N
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
