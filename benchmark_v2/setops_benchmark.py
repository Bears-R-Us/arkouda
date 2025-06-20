import numpy as np
import pytest

import arkouda as ak

OPS = ("intersect", "union", "setdiff", "setxor")
OPS1D = ("intersect1d", "union1d", "setxor1d", "setdiff1d")
TYPES = ("int64", "uint64")


@pytest.mark.skip_correctness_only(True)
@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Segarray_Setops_Small")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_segarr_setops_small(benchmark, op, dtype):
    cfg = ak.get_config()
    full_N = pytest.prob_size * cfg["numLocales"]
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

    nbytes = (
        (seg_a.values.size * seg_a.values.itemsize) + (seg_a.segments.size * seg_a.segments.itemsize)
    ) * 2

    benchmark.extra_info["description"] = "Measures the performance of SegArray setops (small input)"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nbytes / benchmark.stats["mean"]) / 2**30
    )


@pytest.mark.benchmark(group="Setops")
@pytest.mark.parametrize("op", OPS1D)
@pytest.mark.parametrize("dtype", TYPES)
def bench_setops(benchmark, op, dtype):
    cfg = ak.get_config()
    N = 10**4 if pytest.correctness_only else pytest.prob_size * cfg["numLocales"]

    # Always create Arkouda arrays
    if dtype == "int64":
        a_ak = ak.randint(0, 2**32, N, seed=pytest.seed)
        b_ak = ak.randint(0, 2**32, N, seed=pytest.seed)
    elif dtype == "uint64":
        a_ak = ak.randint(0, 2**32, N, seed=pytest.seed, dtype=ak.uint64)
        b_ak = ak.randint(0, 2**32, N, seed=pytest.seed, dtype=ak.uint64)

    if pytest.numpy:
        a = a_ak.to_ndarray()
        b = b_ak.to_ndarray()

        fxn = getattr(np, op)

        def np_op():
            return fxn(a, b)

        benchmark.pedantic(np_op, rounds=pytest.trials)
        numBytes = a.size * a.itemsize * 2
        backend = "NumPy"
    else:
        fxn = getattr(ak, op)

        def ak_op():
            return fxn(a_ak, b_ak)

        benchmark.pedantic(ak_op, rounds=pytest.trials)
        numBytes = a_ak.size * a_ak.itemsize * 2
        backend = "Arkouda"

        if pytest.correctness_only:
            a_np = a_ak.to_ndarray()
            b_np = b_ak.to_ndarray()
            expected = getattr(np, op)(a_np, b_np)
            actual = fxn(a_ak, b_ak).to_ndarray()
            np.testing.assert_array_equal(np.sort(actual), np.sort(expected))

    benchmark.extra_info["description"] = f"Measures the performance of {backend} {op}"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = backend
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (numBytes / benchmark.stats["mean"]) / 2**30
    )
