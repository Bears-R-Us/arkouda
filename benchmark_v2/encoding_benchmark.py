import numpy as np
import pytest

import arkouda as ak

ENCODINGS = ("idna", "ascii")


def generate_string_array(N):
    return ak.random_strings_uniform(1, 16, N, seed=pytest.seed)


def compute_nbytes(a):
    return a.nbytes * a.entry.itemsize


@pytest.mark.skip_correctness_only(True)
@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Strings_EncodeDecode")
@pytest.mark.parametrize("encoding", ENCODINGS)
def bench_encode(benchmark, encoding):
    N = 10**4 if pytest.correctness_only else pytest.prob_size * ak.get_config()["numLocales"]
    a = generate_string_array(N)
    nbytes = compute_nbytes(a)

    def encode_op():
        encoded = a.encode(encoding)  # noqa: F841
        return nbytes

    numBytes = benchmark.pedantic(encode_op, rounds=pytest.trials)

    if pytest.correctness_only:
        input_strs = a.to_ndarray()
        expected = np.array([s.encode(encoding) for s in input_strs])
        np.testing.assert_array_equal(a.encode(encoding).to_ndarray(), expected)

    benchmark.extra_info["description"] = f"Measures the performance of Strings.encode with '{encoding}'"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (numBytes / benchmark.stats["mean"]) / 2**30
    )


@pytest.mark.skip_correctness_only(True)
@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Strings_EncodeDecode")
@pytest.mark.parametrize("encoding", ENCODINGS)
def bench_decode(benchmark, encoding):
    N = 10**4 if pytest.correctness_only else pytest.prob_size * ak.get_config()["numLocales"]
    a = generate_string_array(N)
    encoded = a.encode(encoding)
    nbytes = compute_nbytes(a)

    def decode_op():
        decoded = encoded.decode(encoding)  # noqa: F841
        return nbytes

    numBytes = benchmark.pedantic(decode_op, rounds=pytest.trials)

    if pytest.correctness_only:
        expected = np.array([s for s in a.to_ndarray()])
        result = encoded.decode(encoding).to_ndarray()
        np.testing.assert_array_equal(result, expected)

    benchmark.extra_info["description"] = f"Measures the performance of Strings.decode with '{encoding}'"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (numBytes / benchmark.stats["mean"]) / 2**30
    )
