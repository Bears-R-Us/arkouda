import pytest

import arkouda as ak

OPS = ["and", "or", "shift"]


def _perform_and_binop(a, b):
    return a & b


def _perform_or_binop(a, b):
    return a | b


def _perform_shift_binop(a):
    return a >> 10

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Bigint Bitwise Binops")
@pytest.mark.parametrize("op", OPS)
def bench_ak_bitwise_binops(benchmark, op):
    """
    Measures the performance of bigint bitwise binops
    """
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]

    a1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    a2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    a = ak.bigint_from_uint_arrays([a1, a2], max_bits=pytest.max_bits)
    b1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    b2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    b = ak.bigint_from_uint_arrays([b1, b2], max_bits=pytest.max_bits)

    # bytes per bigint array (N * 16) since it's made of 2 uint64 arrays
    # if max_bits in [0, 64] then they're essentially 1 uint64 array
    nbytes = N * 8 if pytest.max_bits != -1 and pytest.max_bits <= 64 else N * 8 * 2

    if op == "and":
        benchmark.pedantic(_perform_and_binop, args=[a, b], rounds=pytest.trials)
    elif op == "or":
        benchmark.pedantic(_perform_or_binop, args=[a, b], rounds=pytest.trials)
    elif op == "shift":
        benchmark.pedantic(_perform_shift_binop, args=[a], rounds=pytest.trials)

    benchmark.extra_info["description"] = "Measures the performance of bigint bitwise binops"
    benchmark.extra_info["problem_size"] = pytest.prob_size
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nbytes / benchmark.stats["mean"]) / 2**30
    )
    benchmark.extra_info["max_bit"] = pytest.max_bits  # useful when looking at bigint
