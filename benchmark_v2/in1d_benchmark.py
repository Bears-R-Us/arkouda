import arkouda as ak
import pytest

TYPES = ("int64", "uint64", "str")

# Tied to src/In1d.chpl:threshold, which defaults to 2**23
THRESHOLD = 2**23
SIZES = {"MEDIUM": THRESHOLD - 1, "LARGE": THRESHOLD + 1}
MAXSTRLEN = 5

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Arkouda_in1d")
@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("size", SIZES)
def bench_ak_in1d(benchmark, dtype, size):
    """
    Measures the performance of ak.in1d

    """
    if dtype in pytest.dtype:
        cfg = ak.get_config()
        N = pytest.prob_size * cfg["numLocales"]
        s = SIZES[size]

        if dtype == "str":
            a = ak.random_strings_uniform(1, MAXSTRLEN, N)
            b = ak.random_strings_uniform(1, MAXSTRLEN, s)
            nbytes = (a.size * 8 + a.nbytes + b.size * 8 + b.nbytes)
        else:
            a = ak.arange(N) % SIZES["LARGE"]
            b = ak.arange(s)
            nbytes = (a.size * a.itemsize + b.size * b.itemsize)

            if dtype == "uint64":
                a = ak.cast(a, ak.uint64)
                b = ak.cast(b, ak.uint64)

        benchmark.pedantic(ak.in1d, args=(a, b), rounds=pytest.trials)

        benchmark.extra_info["description"] = "Measures the performance of ak.in1d"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2 ** 30)
