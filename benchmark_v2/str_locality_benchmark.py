import pytest

import arkouda as ak

OPS = {
    "Hashing": lambda x: x.hash(),
    "Regex_Search": lambda x: x.contains(r"\d{3,5}\.\d{5,8}", regex=True),
    "Casting": lambda x: ak.cast(x, ak.float64),
    "Scalar_Compare": lambda x: (x == "5.5"),
}

LOCALITY = {"Good", "Poor"}


def _generate_data(loc):
    N = 10**4 if pytest.correctness_only else pytest.prob_size * ak.get_config()["numLocales"]
    prefix = ak.random_strings_uniform(
        minlen=1, maxlen=16, size=N, seed=pytest.seed, characters="numeric"
    )
    if pytest.seed is not None:
        pytest.seed += 1
    suffix = ak.random_strings_uniform(
        minlen=1, maxlen=16, size=N, seed=pytest.seed, characters="numeric"
    )
    random_strings = prefix.stick(suffix, delimiter=".")
    perm = ak.argsort(random_strings.get_lengths())
    sorted_strings = random_strings[perm]
    return random_strings if loc == "Good" else sorted_strings


@pytest.mark.skip_correctness_only(True)
@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="String_Locality")
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("loc", LOCALITY)
def bench_str_locality(benchmark, op, loc):
    data = _generate_data(loc)

    def run():
        OPS[op](data)
        return data.nbytes

    nbytes = benchmark.pedantic(run, rounds=pytest.trials)

    benchmark.extra_info["description"] = (
        f"Benchmark of String locality effects: operation={op}, locality={loc}"
    )
    benchmark.extra_info["problem_size"] = data.size
    benchmark.extra_info["backend"] = "Arkouda"
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nbytes / benchmark.stats["mean"]) / 2**30
    )
