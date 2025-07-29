import pytest

import arkouda as ak

SPLIT_MODES = [
    ("nonregex", "_", False),
    ("regex_literal", "_", True),
    ("regex_pattern", "_+", True),
]


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_Strings_Split")
@pytest.mark.parametrize("label, delim, use_regex", SPLIT_MODES)
def bench_strings_split(benchmark, label, delim, use_regex):
    N = pytest.N

    thirds = [ak.cast(ak.arange(i, N * 3, 3), "str") for i in range(3)]
    thickrange = thirds[0].stick(thirds[1], delimiter="_").stick(thirds[2], delimiter="_")
    nbytes = thickrange.nbytes

    def run():
        thickrange.split(delim, regex=use_regex)
        return nbytes

    num_bytes = benchmark.pedantic(run, rounds=pytest.trials)

    benchmark.extra_info["description"] = f"Performance of Strings.split (mode: {label})"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = "Arkouda"
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (num_bytes / benchmark.stats["mean"]) / 2**30
    )
