import pytest

import arkouda as ak

TYPES = ("int64", "uint64", "float64")
METHODS = ("MergeShuffle",)


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="AK_shuffle")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("method", METHODS)
def bench_shuffle(benchmark, dtype, method):
    """
    Measure rng.shuffle performance on a 1..N array for each dtype in TYPES,
    using arkouda's default RNG and the given shuffle method(s).
    """
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]
    seed = pytest.seed or 0

    # Build 1..N in the desired dtype
    if dtype == "int64":
        pda = ak.arange(1, N + 1, 1, dtype=ak.int64)
    elif dtype == "uint64":
        pda = ak.arange(1, N + 1, 1, dtype=ak.uint64)
    elif dtype == "float64":
        # Note: arange in float64 to follow the "1..N" requirement
        pda = ak.arange(1, N + 1, 1, dtype=ak.float64)

    # Create RNG
    rng = ak.random.default_rng(seed)

    # Compute bytes involved (string arrays report nbytes appropriately)
    nbytes = pda.nbytes

    # Benchmark: shuffle in place
    benchmark.pedantic(
        rng.shuffle,
        args=[pda],
        kwargs={"method": method},
        rounds=pytest.trials,
    )

    # Extra metadata similar to your find benchmark
    benchmark.extra_info["description"] = (
        f"Shuffles a 1..N array with dtype={dtype} using rng.shuffle(method='{method}')."
    )
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nbytes / benchmark.stats["mean"]) / 2**30
    )
