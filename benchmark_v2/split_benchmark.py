import arkouda as ak
import pytest


def _generate_test_data():
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]

    thirds = [ak.cast(ak.arange(i, N * 3, 3), "str") for i in range(3)]
    thickrange = thirds[0].stick(thirds[1], delimiter="_").stick(thirds[2], delimiter="_")
    nbytes = thickrange.nbytes * thickrange.entry.itemsize

    return thickrange, nbytes

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="AK_Flatten")
def bench_split_nonregex(benchmark):
    thickrange, nbytes = _generate_test_data()

    benchmark.pedantic(thickrange.split, args=["_"], rounds=pytest.trials)
    benchmark.extra_info["description"] = "Measures the performance of Strings.split"
    benchmark.extra_info["problem_size"] = pytest.prob_size
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nbytes / benchmark.stats["mean"]) / 2**30
    )

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="AK_Flatten")
def bench_split_regexliteral(benchmark):
    thickrange, nbytes = _generate_test_data()

    benchmark.pedantic(thickrange.split, args=["_"], kwargs={"regex": True}, rounds=pytest.trials)
    benchmark.extra_info["description"] = "Measures the performance of Strings.split"
    benchmark.extra_info["problem_size"] = pytest.prob_size
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nbytes / benchmark.stats["mean"]) / 2**30
    )

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="AK_Flatten")
def bench_split_regexpattern(benchmark):
    thickrange, nbytes = _generate_test_data()

    benchmark.pedantic(thickrange.split, args=["_+"], kwargs={"regex": True}, rounds=pytest.trials)
    benchmark.extra_info["description"] = "Measures the performance of Strings.split"
    benchmark.extra_info["problem_size"] = pytest.prob_size
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nbytes / benchmark.stats["mean"]) / 2**30
    )
