from benchmark_utils import calc_num_bytes
import pytest

import arkouda as ak

TYPES = ("int64", "float64", "bigint")


def create_ak_array(N, dtype):
    if dtype == ak.bigint.name:
        u1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
        u2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
        a = ak.bigint_from_uint_arrays([u1, u2], max_bits=pytest.max_bits)
    else:
        a = ak.randint(0, 2**32, N, dtype=dtype, seed=pytest.seed)
    return a


@pytest.mark.benchmark(group="ArrayTransfer_tondarray")
@pytest.mark.parametrize("dtype", TYPES)
def bench_array_transfer_tondarray(benchmark, dtype):
    if dtype in pytest.dtype:
        N = pytest.N
        a = create_ak_array(N, dtype)
        num_bytes = calc_num_bytes(a)
        ak.client.maxTransferBytes = num_bytes

        def to_nd():
            a.to_ndarray()

        benchmark.pedantic(to_nd, rounds=pytest.trials)

        benchmark.extra_info["description"] = "Measures the performance of pdarray.to_ndarray"
        benchmark.extra_info["problem_size"] = N
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
        benchmark.extra_info["max_bit"] = pytest.max_bits


@pytest.mark.benchmark(group="ArrayTransfer_ak.array")
@pytest.mark.parametrize("dtype", TYPES)
def bench_array_transfer_akarray(benchmark, dtype):
    if dtype in pytest.dtype:
        N = pytest.N
        a = create_ak_array(N, dtype)
        num_bytes = calc_num_bytes(a)
        ak.client.maxTransferBytes = num_bytes
        npa = a.to_ndarray()

        def from_np():
            ak.array(npa, max_bits=pytest.max_bits)

        benchmark.pedantic(from_np, rounds=pytest.trials)

        benchmark.extra_info["description"] = "Measures the performance of ak.array"
        benchmark.extra_info["problem_size"] = N
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
        benchmark.extra_info["max_bit"] = pytest.max_bits
