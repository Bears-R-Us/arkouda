import arkouda as ak
import pytest

TYPES = ("int64", "uint64", "float64", "str")

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
            (nbytes / benchmark.stats["mean"]) / 2 ** 30)
