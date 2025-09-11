import pytest

import arkouda as ak


ENCODINGS = ("idna", "ascii")


def generate_string_array(N):
    return ak.random_strings_uniform(1, 16, N, seed=pytest.seed)


def compute_nbytes(a):
    return a.nbytes * a.entry.itemsize


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Strings_EncodeDecode")
@pytest.mark.parametrize("encoding", ENCODINGS)
def bench_encode(benchmark, encoding):
    N = pytest.N
    a = generate_string_array(N)
    num_bytes = compute_nbytes(a)

    def encode_op():
        encoded = a.encode(encoding)  # noqa: F841

    benchmark.pedantic(encode_op, rounds=pytest.trials)

    benchmark.extra_info["description"] = f"Measures the performance of Strings.encode with '{encoding}'"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="Strings_EncodeDecode")
@pytest.mark.parametrize("encoding", ENCODINGS)
def bench_decode(benchmark, encoding):
    N = pytest.N
    a = generate_string_array(N)
    encoded = a.encode(encoding)
    num_bytes = compute_nbytes(a)

    def decode_op():
        decoded = encoded.decode(encoding)  # noqa: F841

    benchmark.pedantic(decode_op, rounds=pytest.trials)

    benchmark.extra_info["description"] = f"Measures the performance of Strings.decode with '{encoding}'"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
