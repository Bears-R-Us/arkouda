import pytest

import arkouda as ak


OPS = ["and", "or", "shift"]


def generate_bigint_pair(N):
    a1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    a2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    return ak.bigint_from_uint_arrays([a1, a2], max_bits=pytest.max_bits)


def compute_nbytes(N):
    if pytest.max_bits != -1 and pytest.max_bits <= 64:
        return N * 8
    return N * 16


@pytest.mark.benchmark(group="Bigint Bitwise Binops")
@pytest.mark.parametrize("op", OPS)
def bench_bitwise_binops(benchmark, op):
    """
    Measures the performance of bigint bitwise binops.
    """
    N = pytest.N
    num_bytes = compute_nbytes(N)

    a = generate_bigint_pair(N)
    b = generate_bigint_pair(N)

    if op == "and":

        def do_op():
            _ = a & b

    elif op == "or":

        def do_op():
            _ = a | b

    elif op == "shift":

        def do_op():
            _ = a >> 10

    benchmark.pedantic(do_op, rounds=pytest.trials)

    benchmark.extra_info["description"] = "Measures the performance of bigint bitwise binops"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
    benchmark.extra_info["max_bit"] = pytest.max_bits
