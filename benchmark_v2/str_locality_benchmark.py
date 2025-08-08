import pytest

import arkouda as ak

OPS = {
    "Hashing": lambda x: x.hash(),
    "Regex": lambda x: x.contains(r"\d{3,5}\.\d{5,8}", regex=True),
    "Casting": lambda x: ak.cast(x, ak.float64),
    "Comparing": lambda x: (x == "5.5"),
}

LOCALITY = {"good", "poor"}


def _generate_data(loc):
    N = pytest.prob_size * ak.get_config()["numLocales"]
    prefix = ak.random_strings_uniform(
        minlen=1, maxlen=16, size=N, seed=pytest.seed, characters="numeric"
    )
    suffix = ak.random_strings_uniform(
        minlen=1,
        maxlen=16,
        size=N,
        seed=None if pytest.seed is None else pytest.seed + 1,
        characters="numeric",
    )
    random_strings = prefix.stick(suffix, delimiter=".")

    if loc == "good":
        return random_strings
    else:
        #   To simulate poor locality, sort by string lengths.
        return random_strings[ak.argsort(random_strings.get_lengths())]


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
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((nbytes / benchmark.stats["mean"]) / 2**30)
