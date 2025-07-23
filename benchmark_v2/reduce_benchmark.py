import pytest

import arkouda as ak

OPS = ("sum", "prod", "min", "max", "argmin", "argmax")
TYPES = ("int64", "float64")


@pytest.mark.benchmark(group="Reduce")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", TYPES)
def bench_reduce(benchmark, op, dtype):
    if dtype not in pytest.dtype:
        pytest.skip(f"{dtype} not in selected dtypes")

    N = pytest.N

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

    benchmark.pedantic(fxn, rounds=pytest.trials)
    nbytes = a.size * a.itemsize
    benchmark.extra_info["description"] = f"Reduce: {op} ({backend})"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = backend
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((nbytes / benchmark.stats["mean"]) / 2**30)
