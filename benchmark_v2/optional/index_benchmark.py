import pytest

import arkouda as ak
from benchmark_v2.benchmark_utils import calc_num_bytes


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="AK_string_index")
def bench_string_index(benchmark):
    """
    Measure __getitem__ performance for Strings

    """
    N = pytest.N
    ind_frac = 0.5  # Use 50% of the arr as indexing set
    iN = max(1, int(ind_frac * N))

    seed = pytest.seed or 0

    arr = ak.random_strings_uniform(1, 16, N, seed=seed)
    iArr = ak.randint(0, N, iN, seed=seed + 1)

    num_bytes = calc_num_bytes((arr, iArr))

    benchmark.pedantic(
        ak.Strings.__getitem__,
        args=[arr, iArr],
        rounds=pytest.trials,
    )

    benchmark.extra_info["description"] = "Measures the performance of ak.find with all_occurrences=True"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["query_size"] = iN
    benchmark.extra_info["num_bytes"] = num_bytes
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (num_bytes / benchmark.stats["mean"]) / 2**30
    )
