import numpy as np
import pytest

import arkouda as ak

OPS = ("intersect", "union", "setdiff", "setxor")
OPS1D = ("intersect1d", "union1d", "setxor1d", "setdiff1d")
TYPES = ("int64", "uint64")


@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Segarray_Setops")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_segarr_setops(benchmark, op, dtype):
    """
    Measures the performance of segarray setops
    """
    if dtype in pytest.dtype:
        cfg = ak.get_config()
        N = pytest.prob_size * cfg["numLocales"]
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

        benchmark.extra_info["description"] = "Measures the performance of segarray setops"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )


@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="AK_Setops")
@pytest.mark.parametrize("op", OPS1D)
@pytest.mark.parametrize("dtype", TYPES)
def bench_setops(benchmark, op, dtype):
    """
    Measures the performance of arkouda setops
    """
    if dtype in pytest.dtype:
        cfg = ak.get_config()
        N = pytest.prob_size * cfg["numLocales"]
        seed = pytest.seed

        if dtype == "int64":
            a = ak.randint(0, 2**32, N, seed=seed)
            b = ak.randint(0, 2**32, N, seed=seed)
        elif dtype == "uint64":
            a = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
            b = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)

        fxn = getattr(ak, op)
        benchmark.pedantic(fxn, args=(a, b), rounds=pytest.trials)

        nbytes = a.size * a.itemsize * 2
        benchmark.extra_info["description"] = "Measures the performance of arkouda setops"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )


@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Numpy_Setops")
@pytest.mark.parametrize("op", OPS1D)
@pytest.mark.parametrize("dtype", TYPES)
def bench_np_setops(benchmark, op, dtype):
    """
    Measures performance of Numpy setops for comparison
    """
    if pytest.numpy and dtype in pytest.dtype:
        seed = pytest.seed
        N = pytest.prob_size

        if seed is not None:
            np.random.seed(seed)
        if dtype == "int64":
            a = np.random.randint(0, 2**32, N)
            b = np.random.randint(0, 2**32, N)
        elif dtype == "uint64":
            a = np.random.randint(0, 2**32, N, "uint64")
            b = np.random.randint(0, 2**32, N, "uint64")

        fxn = getattr(np, op)
        benchmark.pedantic(fxn, args=(a, b), rounds=pytest.trials)

        nbytes = a.size * a.itemsize * 2
        benchmark.extra_info["description"] = "Measures the performance of numpy setops for comparison"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )
