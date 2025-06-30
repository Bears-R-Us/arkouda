import numpy as np
import pytest

import arkouda as ak

OPS = ("sum", "prod", "min", "max")
TYPES = ("int64", "float64")


@pytest.mark.benchmark(group="Reduce")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_reduce(benchmark, op, dtype):
    if dtype not in pytest.dtype:
        pytest.skip(f"{dtype} not in selected dtypes")

    cfg = ak.get_config()
    N = 10**4 if pytest.correctness_only else pytest.prob_size * cfg["numLocales"]

    # Create test array
    if pytest.random or pytest.seed is not None:
        if dtype == "int64":
            a = ak.randint(1, N, N, seed=pytest.seed)
        elif dtype == "float64":
            a = ak.uniform(N, seed=pytest.seed) + 0.5
    else:
        a = ak.arange(1, N, 1)
        if dtype == "float64":
            a = 1.0 * a

    # Choose backend
    if pytest.numpy:
        a_np = a.to_ndarray()
        fxn = getattr(a_np, op)
        backend = "NumPy"
    else:
        fxn = getattr(a, op)
        backend = "Arkouda"

    def run():
        result = fxn()
        if pytest.correctness_only:
            expected = getattr(a.to_ndarray(), op)()
            if op == "prod":
                np.testing.assert_allclose(result, expected, rtol=1e-10)
            else:
                assert result == expected
        return a.size * a.itemsize

    bytes_processed = benchmark.pedantic(run, rounds=pytest.trials)

    benchmark.extra_info["description"] = f"Reduce: {op} ({backend})"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = backend
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (bytes_processed / benchmark.stats["mean"]) / 2**30
    )
