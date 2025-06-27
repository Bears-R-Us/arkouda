import pytest

import arkouda as ak

SIZES = {"small": 6, "medium": 12, "big": 24}


@pytest.mark.skip_numpy(True)
@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="GroupBySmallStrings")
@pytest.mark.parametrize("strlen_label", SIZES)
def bench_groupby_small_str(benchmark, strlen_label):
    N = 10**4 if pytest.correctness_only else pytest.prob_size * ak.get_config()["numLocales"]
    strlen = SIZES[strlen_label]

    a = ak.random_strings_uniform(1, strlen, N, seed=pytest.seed)
    totalbytes = a.nbytes

    def run():
        _ = ak.GroupBy(a)
        return totalbytes

    bytes_processed = benchmark.pedantic(run, rounds=pytest.trials)

    benchmark.extra_info["description"] = f"GroupBy construction benchmark with {strlen_label} strings"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = "Arkouda"
    benchmark.extra_info["string_length"] = strlen
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (bytes_processed / benchmark.stats["mean"]) / 2**30
    )
