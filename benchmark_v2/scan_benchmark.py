from benchmark_utils import calc_num_bytes
import numpy as np
import pytest

import arkouda as ak


OPS = ("cumsum", "cumprod")
TYPES = ("int64", "float64")


@pytest.mark.benchmark(group="Scan")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_scan(benchmark, op, dtype):
    if dtype not in pytest.dtype:
        pytest.skip(f"{dtype} not in selected dtypes")

    N = pytest.N

    if pytest.numpy:
        if pytest.seed is not None:
            np.random.seed(pytest.seed)
        if pytest.random or pytest.seed is not None:
            if dtype == "int64":
                a = np.random.randint(1, N, N)
            else:
                a = np.random.random(N) + 0.5
        else:
            a = np.arange(1, N + 1, dtype=dtype)
        fxn = getattr(np, op)
        backend = "NumPy"
    else:
        if pytest.random or pytest.seed is not None:
            if dtype == "int64":
                a = ak.randint(1, N, N, seed=pytest.seed)
            else:
                a = ak.uniform(N, seed=pytest.seed) + 0.5
        else:
            a = ak.arange(1, N + 1)
            if dtype == "float64":
                a = 1.0 * a
        fxn = getattr(ak, op)
        backend = "Arkouda"

    benchmark.pedantic(fxn, args=[a], rounds=pytest.trials)
    num_bytes = 2 * calc_num_bytes(a)

    benchmark.extra_info["description"] = f"Scan: {op} using {backend}"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = backend
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
