import pytest

import arkouda as ak

TYPES = ("int64", "float64", "bigint")


def create_ak_array(N, dtype):
    if dtype == ak.bigint.name:
        u1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
        u2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
        a = ak.bigint_from_uint_arrays([u1, u2], max_bits=pytest.max_bits)
        nb = a.size * 8 if pytest.max_bits != -1 and pytest.max_bits <= 64 else a.size * 16
    else:
        a = ak.randint(0, 2**32, N, dtype=dtype, seed=pytest.seed)
        nb = a.size * a.itemsize
    return a, nb


@pytest.mark.benchmark(group="ArrayTransfer_tondarray")
@pytest.mark.parametrize("dtype", TYPES)
def bench_array_transfer_tondarray(benchmark, dtype):
    if dtype in pytest.dtype:
        N = pytest.N
        a, nb = create_ak_array(N, dtype)
        ak.client.maxTransferBytes = nb

        def to_nd():
            a.to_ndarray()
            return nb

        numBytes = benchmark.pedantic(to_nd, rounds=pytest.trials)

        benchmark.extra_info["description"] = "Measures the performance of pdarray.to_ndarray"
        benchmark.extra_info["problem_size"] = N
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((numBytes / benchmark.stats["mean"]) / 2**30)
        benchmark.extra_info["max_bit"] = pytest.max_bits


@pytest.mark.benchmark(group="ArrayTransfer_ak.array")
@pytest.mark.parametrize("dtype", TYPES)
def bench_array_transfer_akarray(benchmark, dtype):
    if dtype in pytest.dtype:
        N = pytest.N
        a, nb = create_ak_array(N, dtype)
        ak.client.maxTransferBytes = nb
        npa = a.to_ndarray()

        def from_np():
            ak.array(npa, max_bits=pytest.max_bits)
            return nb

        numBytes = benchmark.pedantic(from_np, rounds=pytest.trials)

        benchmark.extra_info["description"] = "Measures the performance of ak.array"
        benchmark.extra_info["problem_size"] = N
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((numBytes / benchmark.stats["mean"]) / 2**30)
        benchmark.extra_info["max_bit"] = pytest.max_bits
