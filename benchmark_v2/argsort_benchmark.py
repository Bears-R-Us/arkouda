import numpy as np
import pytest

import arkouda as ak

TYPES = ("int64", "uint64", "float64", "str")

@pytest.mark.benchmark(group="arkouda_argsort")
@pytest.mark.skip_correctness_only(True)
@pytest.mark.parametrize("dtype", TYPES)
def bench_argsort(benchmark, dtype):
    """
    Measure ArgSort performance. Runs for each dtype in TYPES

    Note
    -----
    str dtype is significantly slower than numerics
    """
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]
    if dtype in pytest.dtype:
        if dtype == "int64":
            a = ak.randint(0, 2**32, N, seed=pytest.seed)
            nbytes = a.size * a.itemsize
        elif dtype == "uint64":
            a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
            nbytes = a.size * a.itemsize
        elif dtype == "float64":
            a = ak.randint(0, 1, N, dtype=ak.float64, seed=pytest.seed)
            nbytes = a.size * a.itemsize
        elif dtype == "str":
            a = ak.random_strings_uniform(1, 16, N, seed=pytest.seed)
            nbytes = a.nbytes * a.entry.itemsize

        benchmark.pedantic(ak.argsort, args=[a], rounds=pytest.trials)

        benchmark.extra_info["description"] = "Measures the performance of ak.argsort"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )

@pytest.mark.benchmark(group="numpy_argsort")
@pytest.mark.skip_numpy(False)
@pytest.mark.skip_correctness_only(True)
@pytest.mark.parametrize("dtype", TYPES)
def bench_np_argsort(benchmark, dtype):
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]
    if dtype in pytest.dtype:
        np.random.seed(pytest.seed)
        if dtype == "int64":
            a = np.random.randint(0, 2**32, N)
        elif dtype == "uint64":
            a = np.random.randint(0, 2**32, N, dtype=np.uint64)
        elif dtype == "float64":
            a = np.random.random(N)
        elif dtype == "str":
            a = np.cast["str"](np.random.randint(0, 2**32, N))

    benchmark.pedantic(np.argsort, args=[a], rounds=pytest.trials)

    benchmark.extra_info["description"] = "Measures the performance of np.argsort"
    benchmark.extra_info["problem_size"] = pytest.prob_size
    benchmark.extra_info["average_rate"] = "{:.4f} GiB/sec".format(
        ((a.size * a.itemsize) / benchmark.stats["mean"]) / 2**30
    )


@pytest.mark.skip_correctness_only(False)
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("seed", [pytest.seed])
def check_correctness(dtype, seed):
    N = 10**4
    if dtype == "int64":
        a = ak.randint(0, 2**32, N, seed=seed)
    elif dtype == "uint64":
        a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
    elif dtype == "float64":
        a = ak.randint(0, 1, N, dtype=ak.float64, seed=seed)
    elif dtype == "str":
        a = ak.random_strings_uniform(1, 16, N, seed=seed)

    perm = ak.argsort(a)
    if dtype in ("int64", "uint64", "float64"):
        assert ak.is_sorted(a[perm])
