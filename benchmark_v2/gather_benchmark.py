import numpy as np
import pytest

import arkouda as ak

TYPES = ("int64", "float64", "bool", "str")


def _run_gather(a, i):
    """
    Helper function allowing for the gather to be benchmarked
    """
    return a[i]


@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="AK_Gather")
@pytest.mark.parametrize("dtype", TYPES)
def bench_gather(benchmark, dtype):
    cfg = ak.get_config()
    isize = pytest.prob_size if pytest.idx_size is None else pytest.idx_size
    vsize = pytest.prob_size if pytest.val_size is None else pytest.val_size
    Ni = isize * cfg["numLocales"]
    Nv = vsize * cfg["numLocales"]

    if dtype in pytest.dtype:
        i = ak.randint(0, Nv, Ni, seed=pytest.seed)
        if pytest.seed is not None:
            pytest.seed += 1
        if pytest.random or pytest.seed is not None:
            if dtype == "int64":
                v = ak.randint(0, 2**32, Nv, seed=pytest.seed)
            elif dtype == "float64":
                v = ak.randint(0, 1, Nv, dtype=ak.float64, seed=pytest.seed)
            elif dtype == "bool":
                v = ak.randint(0, 1, Nv, dtype=ak.bool, seed=pytest.seed)
            elif dtype == "str":
                v = ak.random_strings_uniform(1, 16, Nv, seed=pytest.seed)
        else:
            if dtype == "str":
                v = ak.cast(ak.arange(Nv), "str")
            else:
                v = ak.ones(Nv, dtype=dtype)

        c = benchmark.pedantic(_run_gather, args=[v, i], rounds=pytest.trials)

        if dtype == "str":
            offsets_transferred = 3 * c.size * 8
            bytes_transferred = (c.size * 8) + (2 * c.nbytes)
            bytes_per_sec = (offsets_transferred + bytes_transferred) / benchmark.stats["mean"]
        else:
            bytes_per_sec = (c.size * c.itemsize * 3) / benchmark.stats["mean"]

        benchmark.extra_info["description"] = "Measures the performance of Arkouda gather"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["index_size"] = isize
        benchmark.extra_info["value_size"] = vsize
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format((bytes_per_sec / 2**30))


@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="NumPy_Gather")
@pytest.mark.parametrize("dtype", TYPES)
def bench_np_gather(benchmark, dtype):
    if pytest.numpy:
        Ni = pytest.prob_size if pytest.idx_size is None else pytest.idx_size
        Nv = pytest.prob_size if pytest.val_size is None else pytest.val_size
        if pytest.seed is not None:
            np.random.seed(pytest.seed)
        i = np.random.randint(0, Nv, Ni)

        if pytest.random or pytest.seed is not None:
            if dtype == "int64":
                v = np.random.randint(0, 2**32, Nv)
            elif dtype == "float64":
                v = np.random.random(Nv)
            elif dtype == "bool":
                v = np.random.randint(0, 1, Nv, dtype=np.bool)
            elif dtype == "str":
                v = np.array(np.random.randint(0, 2**32, Nv), dtype="str")
        else:
            v = np.ones(Nv, dtype=dtype)

        c = benchmark.pedantic(_run_gather, args=[v, i], rounds=pytest.trials)
        bytes_per_sec = (c.size * c.itemsize * 3) / benchmark.stats["mean"]
        benchmark.extra_info["description"] = "Measures the performance of NumPy gather"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["index_size"] = Ni
        benchmark.extra_info["value_size"] = Nv
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format((bytes_per_sec / 2**30))
