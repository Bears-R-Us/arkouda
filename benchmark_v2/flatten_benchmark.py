import pytest

import arkouda as ak


def generate_thickrange(N):
    thirds = [ak.cast(ak.arange(i, N * 3, 3), "str") for i in range(3)]
    return thirds[0].stick(thirds[1], delimiter="_").stick(thirds[2], delimiter="_")


def compute_nbytes(strings):
    return strings.nbytes * strings.entry.itemsize


@pytest.mark.benchmark(group="AK_Flatten")
@pytest.mark.skip_correctness_only(True)
@pytest.mark.skip_numpy(True)
def bench_flatten_nonregex(benchmark):
    N = 10**4 if pytest.correctness_only else pytest.prob_size * ak.get_config()["numLocales"]
    thickrange = generate_thickrange(N)
    nbytes = compute_nbytes(thickrange)

    def flatten_op():
        _ = thickrange.flatten("_")
        return nbytes

    numBytes = benchmark.pedantic(flatten_op, rounds=pytest.trials)

    benchmark.extra_info["description"] = "Measures the performance of Strings.flatten (non-regex)"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (numBytes / benchmark.stats["mean"]) / 2**30
    )


@pytest.mark.benchmark(group="AK_Flatten")
@pytest.mark.skip_correctness_only(True)
@pytest.mark.skip_numpy(True)
def bench_flatten_regexliteral(benchmark):
    N = 10**4 if pytest.correctness_only else pytest.prob_size * ak.get_config()["numLocales"]
    thickrange = generate_thickrange(N)
    nbytes = compute_nbytes(thickrange)

    def flatten_op():
        _ = thickrange.flatten("_", regex=True)
        return nbytes

    numBytes = benchmark.pedantic(flatten_op, rounds=pytest.trials)

    benchmark.extra_info["description"] = (
        "Measures the performance of Strings.flatten (regex=True, literal)"
    )
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (numBytes / benchmark.stats["mean"]) / 2**30
    )


@pytest.mark.benchmark(group="AK_Flatten")
@pytest.mark.skip_correctness_only(True)
@pytest.mark.skip_numpy(True)
def bench_flatten_regexpattern(benchmark):
    N = 10**4 if pytest.correctness_only else pytest.prob_size * ak.get_config()["numLocales"]
    thickrange = generate_thickrange(N)
    nbytes = compute_nbytes(thickrange)

    def flatten_op():
        _ = thickrange.flatten("_+", regex=True)
        return nbytes

    numBytes = benchmark.pedantic(flatten_op, rounds=pytest.trials)

    benchmark.extra_info["description"] = (
        "Measures the performance of Strings.flatten (regex=True, pattern)"
    )
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (numBytes / benchmark.stats["mean"]) / 2**30
    )
