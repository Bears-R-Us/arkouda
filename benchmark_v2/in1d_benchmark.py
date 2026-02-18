import pytest

import arkouda as ak

TYPES = ("int64", "uint64", "str")
# Tied to src/In1d.chpl:threshold, which defaults to 2**23
# This threshold is used to choose between in1d implementation strategies
THRESHOLD = 2**23
SIZES = {"MEDIUM": THRESHOLD - 1, "LARGE": THRESHOLD + 1}
MAXSTRLEN = 5


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Arkouda_in1d")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("size", SIZES)
def bench_in1d(benchmark, dtype, size):
    """
    Benchmark ak.in1d. Skips if --numpy is used.
    """
    if dtype in pytest.dtype:
        cfg = ak.get_config()
        N = pytest.prob_size * cfg["numLocales"]
        s = SIZES[size]

        if dtype == "str":
            a = ak.random_strings_uniform(1, MAXSTRLEN, N)
            b = ak.random_strings_uniform(1, MAXSTRLEN, s)
            nbytes = a.size * 8 + a.nbytes + b.size * 8 + b.nbytes
        else:
            a = ak.arange(N) % SIZES["LARGE"]
            b = ak.arange(s)
            if dtype == "uint64":
                a = ak.cast(a, ak.uint64)
                b = ak.cast(b, ak.uint64)
            nbytes = a.size * a.itemsize + b.size * b.itemsize

        benchmark.pedantic(ak.in1d, args=[a, b], rounds=pytest.trials)
        benchmark.extra_info["description"] = "in1d benchmark using Arkouda"
        benchmark.extra_info["backend"] = "Arkouda"
        benchmark.extra_info["problem_size"] = N
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2**30
        )
