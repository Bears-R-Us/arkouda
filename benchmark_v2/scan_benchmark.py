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

    cfg = ak.get_config()
    N = 10**4 if pytest.correctness_only else pytest.prob_size * cfg["numLocales"]

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

    def run():
        result = fxn(a)
        if pytest.correctness_only:
            expected = getattr(np, op)(a.to_ndarray() if not pytest.numpy else a)
            np.testing.assert_allclose(result.to_ndarray() if not pytest.numpy else result, expected)
        return a.size * a.itemsize * 2  # input + output

    nbytes = benchmark.pedantic(run, rounds=pytest.trials)

    benchmark.extra_info["description"] = f"Scan: {op} using {backend}"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = backend
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nbytes / benchmark.stats["mean"]) / 2**30
    )
