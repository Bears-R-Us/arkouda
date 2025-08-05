import pytest

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

        def scatter_np_op():
            _run_scatter(c, i, v)
            return i.size * i.itemsize * 3

        numBytes = benchmark.pedantic(scatter_np_op, rounds=pytest.trials)
        backend = "NumPy"
    else:

        def scatter_ak_op():
            return _run_scatter(c_ak, i_ak, v_ak)

        numBytes = benchmark.pedantic(scatter_ak_op, rounds=pytest.trials)
        backend = "Arkouda"

    benchmark.extra_info["description"] = f"Measures the performance of {backend} scatter"
    benchmark.extra_info["backend"] = backend
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["index_size"] = isize
    benchmark.extra_info["value_size"] = vsize
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((numBytes / benchmark.stats["mean"]) / 2**30)
