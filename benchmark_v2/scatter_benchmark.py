import arkouda as ak
import pytest
import numpy as np

TYPES = ("int64", "float64", "bool")


def _run_scatter(a, i, v):
    a[i] = v

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="AK_Scatter")
@pytest.mark.parametrize("dtype", TYPES)
def bench_ak_scatter(benchmark, dtype):
    cfg = ak.get_config()
    isize = pytest.prob_size if pytest.idx_size is None else pytest.idx_size
    vsize = pytest.prob_size if pytest.val_size is None else pytest.val_size
    Ni = isize * cfg["numLocales"]
    Nv = vsize * cfg["numLocales"]

    i = ak.randint(0, Nv, Ni, seed=pytest.seed)
    c = ak.zeros(Nv, dtype=dtype)
    if pytest.seed is not None:
        pytest.seed += 1
    if pytest.random or pytest.seed is not None:
        if dtype == "int64":
            v = ak.randint(0, 2 ** 32, Ni, seed=pytest.seed)
        elif dtype == "float64":
            v = ak.randint(0, 1, Ni, dtype=ak.float64, seed=pytest.seed)
        elif dtype == "bool":
            v = ak.randint(0, 1, Ni, dtype=ak.bool, seed=pytest.seed)
    else:
        v = ak.ones(Ni, dtype=dtype)

    benchmark.pedantic(_run_scatter, args=[c, i, v], rounds=pytest.trials)
    bytes_per_sec = (i.size * i.itemsize * 3) / benchmark.stats["mean"]
    benchmark.extra_info["description"] = "Measures the performance of Arkouda scatter"
    benchmark.extra_info["problem_size"] = pytest.prob_size
    benchmark.extra_info["index_size"] = isize
    benchmark.extra_info["value_size"] = vsize
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (bytes_per_sec / 2 ** 30))

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="NumPy_Scatter")
@pytest.mark.parametrize("dtype", TYPES)
def bench_np_scatter(benchmark, dtype):
    Ni = pytest.prob_size if pytest.idx_size is None else pytest.idx_size
    Nv = pytest.prob_size if pytest.val_size is None else pytest.val_size
    if pytest.numpy:
        if pytest.seed is not None:
            np.random.seed(pytest.seed)
        # Index vector is always random
        i = np.random.randint(0, Nv, Ni)
        c = np.zeros(Nv, dtype=dtype)
        if pytest.random or pytest.seed is not None:
            if dtype == "int64":
                v = np.random.randint(0, 2 ** 32, Ni)
            elif dtype == "float64":
                v = np.random.random(Ni)
        else:
            v = np.ones(Ni, dtype=dtype)

        benchmark.pedantic(_run_scatter, args=[c, i, v], rounds=pytest.trials)
        bytes_per_sec = (i.size * i.itemsize * 3) / benchmark.stats["mean"]
        benchmark.extra_info["description"] = "Measures the performance of numpy scatter"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["index_size"] = Ni
        benchmark.extra_info["value_size"] = Nv
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (bytes_per_sec / 2 ** 30))