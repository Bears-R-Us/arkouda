import pytest

from benchmark_utils import calc_num_bytes

import arkouda as ak


TYPES = ("int64", "float64", "bool")


def _run_scatter(a, i, v):
    a[i] = v
    return i.size * i.itemsize * 3


@pytest.mark.benchmark(group="Scatter")
@pytest.mark.parametrize("dtype", TYPES)
def bench_scatter(benchmark, dtype):
    N = pytest.prob_size
    isize = N if pytest.idx_size is None else pytest.idx_size
    vsize = N if pytest.val_size is None else pytest.val_size
    Ni = isize * pytest.cfg["numLocales"]
    Nv = vsize * pytest.cfg["numLocales"]

    # Generate Arkouda arrays
    i_ak = ak.randint(0, Nv, Ni, seed=pytest.seed)
    c_ak = ak.zeros(Nv, dtype=dtype)
    if pytest.seed is not None:
        pytest.seed += 1

    if pytest.random or pytest.seed is not None:
        if dtype == "int64":
            v_ak = ak.randint(0, 2**32, Ni, seed=pytest.seed)
        elif dtype == "float64":
            v_ak = ak.randint(0, 1, Ni, dtype=ak.float64, seed=pytest.seed)
        elif dtype == "bool":
            v_ak = ak.randint(0, 2, Ni, dtype=ak.bool, seed=pytest.seed)
    else:
        v_ak = ak.ones(Ni, dtype=dtype)

    if pytest.numpy:
        i = i_ak.to_ndarray()
        v = v_ak.to_ndarray()
        c = c_ak.to_ndarray()

        benchmark.pedantic(_run_scatter, args=[c, i, v], rounds=pytest.trials)
        num_bytes = calc_num_bytes((i, v, c))
        backend = "NumPy"
    else:
        benchmark.pedantic(_run_scatter, args=[c_ak, i_ak, v_ak], rounds=pytest.trials)
        num_bytes = calc_num_bytes((c_ak, i_ak, v_ak))
        backend = "Arkouda"

    benchmark.extra_info["description"] = f"Measures the performance of {backend} scatter"
    benchmark.extra_info["backend"] = backend
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["index_size"] = isize
    benchmark.extra_info["value_size"] = vsize
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
