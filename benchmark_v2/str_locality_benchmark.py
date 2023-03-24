import arkouda as ak
import pytest

OPS = {
    "Hashing": lambda x: x.hash(),
    "Regex_Search": lambda x: x.contains(r"\d{3,5}\.\d{5,8}", regex=True),
    "Casting": lambda x: ak.cast(x, ak.float64),
    "Scalar_Compare": lambda x: (x == "5.5")
}

# Good - generates random Strings object with "good" locality
# poor - generates a sorted Strings object with "poor" locality
LOCALITY = {"Good", "Poor"}


RAND_DATA = None
SORT_DATA = None

def _generate_data(loc):
    global RAND_DATA, SORT_DATA
    # early out if already set
    if loc == "Good" and RAND_DATA is not None:
        return RAND_DATA
    if loc == "Poor" and SORT_DATA is not None:
        return SORT_DATA
    N = pytest.prob_size * ak.get_config()["numLocales"]
    prefix = ak.random_strings_uniform(minlen=1, maxlen=16, size=N, seed=pytest.seed, characters="numeric")
    if pytest.seed is not None:
        pytest.seed += 1
    suffix = ak.random_strings_uniform(minlen=1, maxlen=16, size=N, seed=pytest.seed, characters="numeric")
    random_strings = prefix.stick(suffix, delimiter=".")
    RAND_DATA = random_strings

    if loc == "Poor":
        perm = ak.argsort(random_strings.get_lengths())
        sorted_strings = random_strings[perm]
        SORT_DATA = sorted_strings
        return SORT_DATA

    return RAND_DATA # Default return for good locality


@pytest.mark.benchmark(group="String_Locality")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("loc", LOCALITY)
def bench_str_locality(benchmark, op, loc):
    data = _generate_data(loc)
    benchmark.pedantic(OPS[op], args=[data], rounds=pytest.trials)

    benchmark.extra_info["description"] = "Measure the performance of various string operations on " \
                                          "strings with good locality (random) and poor locality (sorted)."
    benchmark.extra_info["problem_size"] = pytest.prob_size
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (data.nbytes / benchmark.stats["mean"]) / 2 ** 30)
    benchmark.extra_info["str_name"] = data.name