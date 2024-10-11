import arkouda as ak
import pytest

import numpy as np

TYPES = ["int64", "uint64", "float64", "str"]
NUM_ARR = [1, 2, 8, 16]

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_CoArgSort")
@pytest.mark.parametrize("numArrays", NUM_ARR)
@pytest.mark.parametrize("dtype", TYPES)
def bench_coargsort(benchmark, dtype, numArrays):
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]
    if dtype in pytest.dtype:
        if pytest.seed is None:
            seeds = [None for _ in range(numArrays)]
        else:
            seeds = [pytest.seed + i for i in range(numArrays)]
        if dtype == "int64":
            arrs = [ak.randint(0, 2**32, N // numArrays, seed=s) for s in seeds]
            nbytes = sum(a.size * a.itemsize for a in arrs)
        elif dtype == "uint64":
            arrs = [ak.randint(0, 2**32, N // numArrays, dtype=ak.uint64, seed=s) for s in seeds]
            nbytes = sum(a.size * a.itemsize for a in arrs)
        elif dtype == "float64":
            arrs = [ak.randint(0, 1, N // numArrays, dtype=ak.float64, seed=s) for s in seeds]
            nbytes = sum(a.size * a.itemsize for a in arrs)
        elif dtype == "str":
            arrs = [ak.random_strings_uniform(1, 16, N // numArrays, seed=s) for s in seeds]
            nbytes = sum(a.nbytes * a.entry.itemsize for a in arrs)

        benchmark.pedantic(ak.coargsort, args=[arrs], rounds=pytest.trials)
        benchmark.extra_info["description"] = "Measures the performance of ak.coargsort"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="NumPy_CoArgSort")
@pytest.mark.parametrize("numArrays", NUM_ARR)
@pytest.mark.parametrize("dtype", TYPES)
def bench_coargsort_numpy(benchmark, dtype, numArrays):
    cfg = ak.get_config()
    if pytest.numpy and dtype in pytest.dtype:
        if pytest.seed is not None:
            np.random.seed(pytest.seed)
        if dtype == "int64":
            arrs = [
                np.random.randint(0, 2**32, pytest.prob_size // numArrays) for _ in range(numArrays)
            ]
        elif dtype == "uint64":
            arrs = [
                np.random.randint(0, 2**32, pytest.prob_size // numArrays, dtype=np.uint64)
                for _ in range(numArrays)
            ]
        elif dtype == "float64":
            arrs = [np.random.random(pytest.prob_size // numArrays) for _ in range(numArrays)]
        elif dtype == "str":
            arrs = [
                np.cast["str"](np.random.randint(0, 2**32, pytest.prob_size // numArrays))
                for _ in range(numArrays)
            ]

        nbytes = sum(a.size * a.itemsize for a in arrs)
        benchmark.pedantic(np.lexsort, args=[arrs], rounds=pytest.trials)
        benchmark.extra_info["description"] = "Measures the performance of np.lexsort for comparison"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )
