import arkouda as ak
import pytest

# stores parameters to pass to str.contains
SEARCHES = {
    "Non_Regex": ["1 string 1", False],
    "Regex_Literal": ["1 string 1", True],
    "Regex_Pattern": ["\\d string \\d", True]
}

@pytest.mark.skip_correctness_only(True)
@pytest.mark.parametrize("s", SEARCHES)
def bench_substring_search(benchmark, s):
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]

    start = ak.random_strings_uniform(minlen=1, maxlen=8, size=N, seed=pytest.seed)
    end = ak.random_strings_uniform(minlen=1, maxlen=8, size=N, seed=pytest.seed)

    # each string in test_substring contains '1 string 1' with random strings before and after
    test_substring = start.stick(end, delimiter="1 string 1")
    nbytes = test_substring.nbytes * test_substring.entry.itemsize

    benchmark.pedantic(test_substring.contains, args=SEARCHES[s], rounds=pytest.trials)

    benchmark.extra_info["description"] = "Measure the performance of regex and non-regex substring searches."
    benchmark.extra_info["problem_size"] = pytest.prob_size
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nbytes / benchmark.stats["mean"]) / 2 ** 30)