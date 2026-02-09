import numpy as np
import pytest

from benchmark_utils import calc_num_bytes

import arkouda as ak


OPS = ("intersect", "union", "setdiff", "setxor")
OPS1D = ("intersect1d", "union1d", "setxor1d", "setdiff1d")
TYPES = ("int64", "uint64")


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Segarray_Setops_Small")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_segarr_setops_small(benchmark, op, dtype):
    full_N = pytest.N
    N = max(full_N // 100, 10**6)
    seed = pytest.seed

    if dtype == "int64":
        a = ak.randint(0, 2**32, N, seed=seed)
        b = ak.randint(0, 2**32, N, seed=seed)
        seg_a = ak.SegArray(ak.array([0, len(a)]), ak.concatenate([a, b]))
        c = ak.randint(0, 2**32, N, seed=seed)
        d = ak.randint(0, 2**32, N, seed=seed)
        seg_b = ak.SegArray(ak.array([0, len(c)]), ak.concatenate([c, d]))
    elif dtype == "uint64":
        a = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
        b = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
        seg_a = ak.SegArray(ak.array([0, len(a)]), ak.concatenate([a, b]))
        c = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
        d = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
        seg_b = ak.SegArray(ak.array([0, len(c)]), ak.concatenate([c, d]))

    fxn = getattr(seg_a, op)
    benchmark.pedantic(fxn, args=[seg_b], rounds=pytest.trials)

    num_bytes = calc_num_bytes((seg_a, seg_b))

    benchmark.extra_info["description"] = "Measures the performance of SegArray setops (small input)"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)


@pytest.mark.benchmark(group="Setops")
@pytest.mark.parametrize("op", OPS1D)
@pytest.mark.parametrize("dtype", TYPES)
def bench_setops(benchmark, op, dtype):
    N = pytest.N

    # Always create Arkouda arrays
    a_ak = ak.randint(0, 2**32, N, seed=pytest.seed, dtype=dtype)
    b_ak = ak.randint(0, 2**32, N, seed=pytest.seed, dtype=dtype)

    if pytest.numpy:
        a = a_ak.to_ndarray()
        b = b_ak.to_ndarray()

        fxn = getattr(np, op)

        def np_op():
            return fxn(a, b)

        benchmark.pedantic(np_op, rounds=pytest.trials)
        num_bytes = calc_num_bytes((a, b)) if pytest.numpy else calc_num_bytes((a_ak, b_ak))
        backend = "NumPy"
    else:
        fxn = getattr(ak, op)

        def ak_op():
            return fxn(a_ak, b_ak)

        benchmark.pedantic(ak_op, rounds=pytest.trials)
        num_bytes = calc_num_bytes((a_ak, b_ak))
        backend = "Arkouda"

    benchmark.extra_info["description"] = f"Measures the performance of {backend} {op}"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = backend
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
