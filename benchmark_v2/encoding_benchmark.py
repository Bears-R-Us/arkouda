import arkouda as ak
import pytest

ENCODINGS = ("idna", "ascii")


@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Strings_EncodeDecode")
@pytest.mark.parametrize("encoding", ENCODINGS)
def bench_encode(benchmark, encoding):
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]
    if encoding in pytest.encoding:
        a = ak.random_strings_uniform(1, 16, N, seed=pytest.seed)
        nbytes = a.nbytes * a.entry.itemsize

        benchmark.pedantic(a.encode, args=[encoding], rounds=pytest.trials)
        benchmark.extra_info["description"] = "Measures the performance of Strings.encode"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2 ** 30)

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Strings_EncodeDecode")
@pytest.mark.parametrize("encoding", ENCODINGS)
def bench_decode(benchmark, encoding):
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]
    if encoding in pytest.encoding:
        a = ak.random_strings_uniform(1, 16, N, seed=pytest.seed)
        nbytes = a.nbytes * a.entry.itemsize

        benchmark.pedantic(a.decode, args=[encoding], rounds=pytest.trials)
        benchmark.extra_info["description"] = "Measures the performance of Strings.decode"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nbytes / benchmark.stats["mean"]) / 2 ** 30)
