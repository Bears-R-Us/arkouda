import arkouda as ak
import pytest

@pytest.mark.benchmark(group="BigInt_Conversion")
def bench_to_bigint(benchmark):
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]

    a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    b = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)

    # bytes per bigint array (N * 16) since it's made of 2 uint64 arrays
    # if max_bits in [0, 64] then they're essentially 1 uint64 array
    tot_bytes = N * 8 if pytest.max_bits != -1 and pytest.max_bits <= 64 else N * 8 * 2

    benchmark.pedantic(
        ak.bigint_from_uint_arrays,
        args=[[a, b]],
        kwargs={"max_bits": pytest.max_bits},
        rounds=pytest.trials,
    )

@pytest.mark.benchmark(group="BigInt_Conversion")
def bench_from_bigint(benchmark):
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]

    # bytes per bigint array (N * 16) since it's made of 2 uint64 arrays
    # if max_bits in [0, 64] then they're essentially 1 uint64 array
    tot_bytes = N * 8 if pytest.max_bits != -1 and pytest.max_bits <= 64 else N * 8 * 2

    a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    b = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    ba = ak.bigint_from_uint_arrays([a, b], max_bits=pytest.max_bits)
    benchmark.pedantic(ba.bigint_to_uint_arrays, rounds=pytest.trials)
