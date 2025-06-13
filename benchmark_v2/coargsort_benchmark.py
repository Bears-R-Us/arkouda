import numpy as np
import pytest

import arkouda as ak

TYPES = ["int64", "uint64", "float64", "str"]
NUM_ARR = [1, 2, 8, 16]


@pytest.mark.benchmark(group="Arkouda_CoArgSort")
@pytest.mark.parametrize("numArrays", NUM_ARR)
@pytest.mark.parametrize("dtype", TYPES)
def bench_coargsort(benchmark, dtype, numArrays):
    cfg = ak.get_config()
    N = 10**4 if pytest.correctness_only else pytest.prob_size * cfg["numLocales"]

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

        nbytes = (
            sum(a.size * a.itemsize for a in arrs)
            if dtype != "str"
            else sum(a.nbytes * a.entry.itemsize for a in arrs)
        )

        if pytest.numpy:
            arrs = [a.to_ndarray() for a in arrs]
            func = np.lexsort
        else:
            func = ak.coargsort

        result = benchmark.pedantic(func, args=[arrs], rounds=pytest.trials)

        if pytest.correctness_only and not pytest.numpy and dtype != "str":
            arrs_np = [a.to_ndarray() for a in arrs]
            #   np.lexsort sorts the arrays in reverse key order from ak.coargsort.
            arrs_np.reverse()
            expected = np.lexsort(arrs_np)
            np.testing.assert_array_equal(result.to_ndarray(), expected)

        benchmark.extra_info["description"] = "Measures the performance of ak.coargsort"
        benchmark.extra_info["problem_size"] = N
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )
