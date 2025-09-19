from benchmark_utils import calc_num_bytes
import pytest

import arkouda as ak


SEARCHES = {
    "Non_Regex": ["1 string 1", False],
    "Regex_Literal": ["1 string 1", True],
    "Regex_Pattern": ["\\d string \\d", True],
}


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_Strings_SubstringSearch")
@pytest.mark.parametrize("search_string,use_regex", SEARCHES.values(), ids=list(SEARCHES.keys()))
def bench_strings_contains(benchmark, search_string, use_regex):
    N = pytest.N
    seed = pytest.seed

    start = ak.random_strings_uniform(minlen=1, maxlen=8, size=N, seed=seed)
    end = ak.random_strings_uniform(minlen=1, maxlen=8, size=N, seed=seed)

    # each string in test_substring contains '1 string 1' with random strings before and after
    a = start.stick(end, delimiter="1 string 1")

    num_bytes = calc_num_bytes(a)

    benchmark.pedantic(
        a.contains, args=[search_string], kwargs={"regex": use_regex}, rounds=pytest.trials
    )
    benchmark.extra_info["description"] = (
        f"Benchmark for Strings.contains with regex={use_regex} and search_string={search_string}"
    )
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = "Arkouda"
    benchmark.extra_info["search_string"] = search_string
    benchmark.extra_info["regex"] = use_regex
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
