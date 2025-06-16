import numpy as np
import pytest

import arkouda as ak

TYPES = ("int64", "float64", "bool", "str")


def _run_gather(a, i):
    return a[i]


def compute_transfer_bytes(result, dtype):
    if dtype == "str":
        offsets = 3 * result.size * 8
        buffers = (result.size * 8) + (2 * result.nbytes)
        return offsets + buffers
    else:
        return result.size * result.itemsize * 3


@pytest.mark.benchmark(group="Gather")
@pytest.mark.parametrize("dtype", TYPES)
def bench_gather(benchmark, dtype):
    cfg = ak.get_config()
    N = 10**4 if pytest.correctness_only else pytest.prob_size
    isize = N if pytest.idx_size is None else pytest.idx_size
    vsize = N if pytest.val_size is None else pytest.val_size
    Ni = isize * cfg["numLocales"]
    Nv = vsize * cfg["numLocales"]

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
            c = _run_gather(v, i)
            return compute_transfer_bytes(c, dtype)

        numBytes = benchmark.pedantic(gather_np_op, rounds=pytest.trials)
        backend = "NumPy"
    else:

        def gather_ak_op():
            c = _run_gather(v_ak, i_ak)
            return compute_transfer_bytes(c, dtype)

        numBytes = benchmark.pedantic(gather_ak_op, rounds=pytest.trials)
        backend = "Arkouda"

        # Correctness check: compare against NumPy if enabled
        if pytest.correctness_only:
            i_np = i_ak.to_ndarray()
            v_np = v_ak.to_ndarray()
            expected = v_np[i_np]
            result = _run_gather(v_ak, i_ak).to_ndarray()
            np.testing.assert_array_equal(result, expected)

    benchmark.extra_info["description"] = f"Measures the performance of {backend} gather"
    benchmark.extra_info["backend"] = backend
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["index_size"] = isize
    benchmark.extra_info["value_size"] = vsize
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (numBytes / benchmark.stats["mean"]) / 2**30
    )
