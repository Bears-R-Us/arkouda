import pytest

from benchmark_utils import calc_num_bytes

import arkouda as ak


SIZES = {"small": 6, "medium": 12, "big": 24}


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="GroupBySmallStrings")
@pytest.mark.parametrize("strlen_label", SIZES)
def bench_groupby_small_str(benchmark, strlen_label):
    N = pytest.N
    strlen = SIZES[strlen_label]

    a = ak.random_strings_uniform(1, strlen, N, seed=pytest.seed)
    num_bytes = calc_num_bytes(a)
    benchmark.pedantic(ak.GroupBy, args=[a], rounds=pytest.trials)

    benchmark.extra_info["description"] = f"GroupBy construction benchmark with {strlen_label} strings"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = "Arkouda"
    benchmark.extra_info["string_length"] = strlen
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
