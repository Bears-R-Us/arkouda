from benchmark_utils import calc_num_bytes
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
        N = pytest.N
        s = SIZES[size]

        if dtype == "str":
            a = ak.random_strings_uniform(1, MAXSTRLEN, N)
            b = ak.random_strings_uniform(1, MAXSTRLEN, s)
            num_bytes = calc_num_bytes((a, b))
        else:
            a = ak.arange(N) % SIZES["LARGE"]
            b = ak.arange(s)
            if dtype == "uint64":
                a = ak.cast(a, ak.uint64)
                b = ak.cast(b, ak.uint64)
            num_bytes = calc_num_bytes((a, b))

        benchmark.pedantic(ak.in1d, args=[a, b], rounds=pytest.trials)
        benchmark.extra_info["description"] = "in1d benchmark using Arkouda"
        benchmark.extra_info["backend"] = "Arkouda"
        benchmark.extra_info["problem_size"] = N
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
