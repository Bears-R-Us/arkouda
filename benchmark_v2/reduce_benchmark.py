import numpy as np
import arkouda as ak
import pytest

OPS = ("sum", "prod", "min", "max")
TYPES = ("int64", "float64")

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_Reduce")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_ak_reduce(benchmark, op, dtype):
    if dtype in pytest.dtype:
        cfg = ak.get_config()
        N = pytest.prob_size * cfg["numLocales"]
        if pytest.random or pytest.seed is not None:
            if dtype == "int64":
                a = ak.randint(1, N, N, seed=pytest.seed)
            elif dtype == "float64":
                a = ak.uniform(N, seed=pytest.seed) + 0.5
        else:
            a = ak.arange(1, N, 1)
            if dtype == "float64":
                a = 1.0 * a

        fxn = getattr(a, op)
        benchmark.pedantic(fxn, rounds=pytest.trials)

        nbytes = a.size * a.itemsize
        benchmark.extra_info["description"] = "Measures performance of ak reduce functions."
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2 ** 30
        )

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Numpy_Reduce")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_np_reduce(benchmark, op, dtype):
    if pytest.numpy and dtype in pytest.dtype:
        N = pytest.prob_size
        if pytest.seed is not None:
            np.random.seed(pytest.seed)

        if pytest.random or pytest.seed is not None:
            if dtype == "int64":
                a = np.random.randint(1, N, N)
            elif dtype == "float64":
                a = np.random.random(N) + 0.5
        else:
            a = np.arange(1, N, 1, dtype=dtype)

        fxn = getattr(a, op)
        benchmark.pedantic(fxn, rounds=pytest.trials)

        nbytes = a.size * a.itemsize
        benchmark.extra_info["description"] = "Measures performance of numpy reduce functions for comparison"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2 ** 30
        )
