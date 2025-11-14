import pytest

from benchmark_utils import calc_num_bytes

import arkouda as ak


TYPES = ("int64", "float64", "bool", "str")


def _run_gather(a, i):
    return a[i]


@pytest.mark.benchmark(group="Gather")
@pytest.mark.parametrize("dtype", TYPES)
def bench_gather(benchmark, dtype):
    N = pytest.prob_size
    isize = N if pytest.idx_size is None else pytest.idx_size
    vsize = N if pytest.val_size is None else pytest.val_size
    Ni = isize * pytest.cfg["numLocales"]
    Nv = vsize * pytest.cfg["numLocales"]

    i_ak = ak.randint(0, Nv, Ni, seed=pytest.seed)
    if pytest.seed is not None:
        pytest.seed += 1

    if pytest.random or pytest.seed is not None:
        if dtype == "int64":
            v_ak = ak.randint(0, 2**32, Nv, seed=pytest.seed)
        elif dtype == "float64":
            v_ak = ak.randint(0, 1, Nv, dtype=ak.float64, seed=pytest.seed)
        elif dtype == "bool":
            v_ak = ak.randint(0, 2, Nv, dtype=ak.bool, seed=pytest.seed)
        elif dtype == "str":
            v_ak = ak.random_strings_uniform(1, 16, Nv, seed=pytest.seed)
    else:
        if dtype == "str":
            v_ak = ak.cast(ak.arange(Nv), "str")
        else:
            v_ak = ak.ones(Nv, dtype=dtype)

    if pytest.numpy:
        i = i_ak.to_ndarray()
        v = v_ak.to_ndarray()

        def gather_np_op():
            return _run_gather(v, i)

        result = benchmark.pedantic(gather_np_op, rounds=pytest.trials)
        num_bytes = calc_num_bytes(i) + 2 * calc_num_bytes(result)
        backend = "NumPy"
    else:

        def gather_ak_op():
            return _run_gather(v_ak, i_ak)

        result = benchmark.pedantic(gather_ak_op, rounds=pytest.trials)
        #   To compute the data transfer bytes, add the size of the index `i_ak`,
        #   the size of `result`,
        #   and the portion of v that is indexed into which should have the approx. same size as `result`
        num_bytes = calc_num_bytes(i_ak) + 2 * calc_num_bytes(result)
        backend = "Arkouda"

    benchmark.extra_info["description"] = f"Measures the performance of {backend} gather"
    benchmark.extra_info["backend"] = backend
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["index_size"] = isize
    benchmark.extra_info["value_size"] = vsize
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
