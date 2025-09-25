from benchmark_utils import calc_num_bytes
import pytest

import arkouda as ak


TYPES = ("int64", "uint64", "float64", "str")


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="AK_find")
@pytest.mark.parametrize("dtype", TYPES)
def bench_find(benchmark, dtype):
    """
    Measure ak.find performance with all_occurrences=True.
    Runs for each dtype in TYPES.

    """
    N = pytest.N
    query_frac = 0.01  # Use 1% of the space as query set
    qN = max(1, int(query_frac * N))

    seed = pytest.seed or 0

    if dtype == "int64":
        space = ak.randint(0, 2**32, N, seed=seed)
    elif dtype == "uint64":
        space = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
    elif dtype == "float64":
        space = ak.randint(0, 1, N, dtype=ak.float64, seed=seed)
    elif dtype == "str":
        space = ak.random_strings_uniform(1, 16, N, seed=seed)

    # Select query values from the space to guarantee matches
    query = space[ak.randint(0, N, qN, seed=seed + 1)]

    num_bytes = calc_num_bytes((space, query))

    benchmark.pedantic(
        ak.find,
        args=[query, space],
        kwargs={"all_occurrences": True},
        rounds=pytest.trials,
    )

    benchmark.extra_info["description"] = "Measures the performance of ak.find with all_occurrences=True"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["query_size"] = qN
    benchmark.extra_info["num_bytes"] = num_bytes
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (num_bytes / benchmark.stats["mean"]) / 2**30
    )
