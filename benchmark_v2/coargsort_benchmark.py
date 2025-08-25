from benchmark_utils import calc_num_bytes
import numpy as np
import pytest

import arkouda as ak

TYPES = ["int64", "uint64", "float64", "str"]
NUM_ARR = [1, 2, 8, 16]


@pytest.mark.benchmark(group="Arkouda_CoArgSort")
@pytest.mark.parametrize("numArrays", NUM_ARR)
@pytest.mark.parametrize("dtype", TYPES)
def bench_coargsort(benchmark, dtype, numArrays):
    N = pytest.N

    if dtype in pytest.dtype:
        if pytest.seed is None:
            seeds = [None for _ in range(numArrays)]
        else:
            seeds = [pytest.seed + i for i in range(numArrays)]

        # Generate Arkouda arrays
        if dtype == "int64":
            arrs = [ak.randint(0, 2**32, N // numArrays, seed=s) for s in seeds]
        elif dtype == "uint64":
            arrs = [ak.randint(0, 2**32, N // numArrays, dtype=ak.uint64, seed=s) for s in seeds]
        elif dtype == "float64":
            arrs = [ak.randint(0, 1, N // numArrays, dtype=ak.float64, seed=s) for s in seeds]
        elif dtype == "str":
            arrs = [ak.random_strings_uniform(1, 16, N // numArrays, seed=s) for s in seeds]

        num_bytes = calc_num_bytes(arrs)

        if pytest.numpy:
            arrs = [a.to_ndarray() for a in arrs]
            func = np.lexsort
        else:
            func = ak.coargsort

        benchmark.pedantic(func, args=[arrs], rounds=pytest.trials)

        benchmark.extra_info["description"] = "Measures the performance of ak.coargsort"
        benchmark.extra_info["problem_size"] = N
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
