import arkouda as ak
import pytest


@pytest.mark.benchmark(group="Inner Join")
@pytest.mark.parametrize("algo", ["jaro", "jaccard"])
def bench_string_matching(benchmark, algo):
    queryset = ak.random_strings_uniform(2**8, 2**10, pytest.prob_size)
    dataset = queryset[ak.randint(0, pytest.prob_size, pytest.prob_size)]

    benchmark.pedantic(ak.string_match, args=(queryset, dataset, algo), rounds=pytest.trials)
