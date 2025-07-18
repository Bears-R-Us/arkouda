import pytest

import arkouda as ak

OPS = ("intersect1d", "union1d", "setxor1d", "setdiff1d")
DTYPES = ("int64", "uint64")


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="SetOps_MultiArray")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
def bench_setops_multiarray(benchmark, op, dtype):
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]

    seed = pytest.seed or 0
    if dtype == "int64":
        a = ak.randint(0, 2**32, N, seed=seed)
        b = ak.randint(0, 2**32, N, seed=seed + 1)
        c = ak.randint(0, 2**32, N, seed=seed + 2)
        d = ak.randint(0, 2**32, N, seed=seed + 3)
    elif dtype == "uint64":
        a = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
        b = ak.randint(0, 2**32, N, seed=seed + 1, dtype=ak.uint64)
        c = ak.randint(0, 2**32, N, seed=seed + 2, dtype=ak.uint64)
        d = ak.randint(0, 2**32, N, seed=seed + 3, dtype=ak.uint64)
    else:
        raise TypeError("bench_setops_multiarray currently only supports int64 adn uint64 dtypes.")

    fxn = getattr(ak, op)
    benchmark.pedantic(fxn, args=([a, b], [c, d]), rounds=pytest.trials)
    bytes_processed = 4 * a.size * a.itemsize

    benchmark.extra_info["description"] = f"Multiarray set operation: {op} on dtype={dtype}"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = "Arkouda"
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (bytes_processed / benchmark.stats["mean"]) / 2**30
    )
