import numpy as np
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
    Benchmark ak.in1d with support for --correctness_only. Skips if --numpy is used.
    """
    if dtype in pytest.dtype:
        cfg = ak.get_config()
        N = 10**4 if pytest.correctness_only else pytest.prob_size * cfg["numLocales"]
        s = min(SIZES[size], 10**4) if pytest.correctness_only else SIZES[size]

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

        def run():
            result = ak.in1d(a, b)
            if pytest.correctness_only:
                expected = np.isin(a.to_ndarray(), b.to_ndarray())
                np.testing.assert_array_equal(result.to_ndarray(), expected)
            return nbytes

        bytes_processed = benchmark.pedantic(run, rounds=pytest.trials)
        benchmark.extra_info["description"] = "in1d benchmark using Arkouda"
        benchmark.extra_info["backend"] = "Arkouda"
        benchmark.extra_info["problem_size"] = N
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (bytes_processed / benchmark.stats["mean"]) / 2**30
        )
