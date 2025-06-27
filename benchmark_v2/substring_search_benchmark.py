import pytest

import arkouda as ak

# stores parameters to pass to str.contains
SEARCHES = {
    "Non_Regex": ["1 string 1", False],
    "Regex_Literal": ["1 string 1", True],
    "Regex_Pattern": ["\\d string \\d", True],
}


@pytest.mark.skip_correctness_only(True)
@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_Strings_SubstringSearch")
@pytest.mark.parametrize("arg", SEARCHES)
def bench_strings_contains(benchmark, arg):
    N = pytest.prob_size * ak.get_config()["numLocales"]
    seed = pytest.seed

    search_string = arg[0]
    use_regex = bool(arg[1])

    start = ak.random_strings_uniform(minlen=1, maxlen=8, size=N, seed=seed)
    end = ak.random_strings_uniform(minlen=1, maxlen=8, size=N, seed=seed)

    # each string in test_substring contains '1 string 1' with random strings before and after
    a = start.stick(end, delimiter="1 string 1")

    def run():
        a.contains(search_string, regex=bool(use_regex))
        return a.nbytes

    num_bytes = benchmark.pedantic(run, rounds=pytest.trials)
    benchmark.extra_info["description"] = f"Measures substring search performance (regex={use_regex})"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = "Arkouda"
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (num_bytes / benchmark.stats["mean"]) / 2**30
    )
