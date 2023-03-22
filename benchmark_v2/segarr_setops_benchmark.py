import arkouda as ak
import pytest

OPS = ("intersect", "union", "setdiff", "setxor")
TYPES = (
    "int64",
    "uint64"
)


@pytest.mark.benchmark(group="Segarray_Setops")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_ak_setops(benchmark, op, dtype):
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
            seg_a = ak.segarray(ak.array([0, len(a)]), ak.concatenate([a, b]))
            c = ak.randint(0, 2**32, N, seed=seed)
            d = ak.randint(0, 2**32, N, seed=seed)
            seg_b = ak.segarray(ak.array([0, len(c)]), ak.concatenate([c, d]))
        elif dtype == "uint64":
            a = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
            b = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
            seg_a = ak.segarray(ak.array([0, len(a)]), ak.concatenate([a, b]))
            c = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
            d = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
            seg_b = ak.segarray(ak.array([0, len(c)]), ak.concatenate([c, d]))

        fxn = getattr(seg_a, op)
        benchmark.pedantic(fxn, args=[seg_b], rounds=pytest.trials)

        nbytes = a.size * a.itemsize * 2

        benchmark.extra_info["description"] = "Measures the performance of segarray setops"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2 ** 30)
