import arkouda as ak
import pytest

TYPES = ("int64", "float64", "bigint")

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="ArrayTransfer_tondarray")
@pytest.mark.parametrize("dtype", TYPES)
def bench_array_transfer_tondarray(benchmark, dtype):
    if dtype in pytest.dtype:
        if dtype == ak.bigint.name:
            u1 = ak.randint(0, 2 ** 32, pytest.prob_size, dtype=ak.uint64, seed=pytest.seed)
            u2 = ak.randint(0, 2 ** 32, pytest.prob_size, dtype=ak.uint64, seed=pytest.seed)
            a = ak.bigint_from_uint_arrays([u1, u2], max_bits=pytest.max_bits)
            # bytes per bigint array (N * 16) since it's made of 2 uint64 arrays
            # if max_bits in [0, 64] then they're essentially 1 uint64 array
            nb = a.size * 8 if pytest.max_bits != -1 and pytest.max_bits <= 64 else a.size * 8 * 2
            ak.client.maxTransferBytes = nb
        else:
            a = ak.randint(0, 2 ** 32, pytest.prob_size, dtype=dtype, seed=pytest.seed)
            nb = a.size * a.itemsize
            ak.client.maxTransferBytes = nb

        benchmark.pedantic(a.to_ndarray, rounds=pytest.trials)
        benchmark.extra_info["description"] = "Measures the performance of pdarray.to_ndarray"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nb / benchmark.stats["mean"]) / 2 ** 30)
        benchmark.extra_info["max_bit"] = pytest.max_bits  # useful when looking at bigint

@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="ArrayTransfer_ak.array")
@pytest.mark.parametrize("dtype", TYPES)
def bench_array_transfer_akarray(benchmark, dtype):
    if dtype in pytest.dtype:
        if dtype == ak.bigint.name:
            u1 = ak.randint(0, 2 ** 32, pytest.prob_size, dtype=ak.uint64, seed=pytest.seed)
            u2 = ak.randint(0, 2 ** 32, pytest.prob_size, dtype=ak.uint64, seed=pytest.seed)
            a = ak.bigint_from_uint_arrays([u1, u2], max_bits=pytest.max_bits)
            # bytes per bigint array (N * 16) since it's made of 2 uint64 arrays
            # if max_bits in [0, 64] then they're essentially 1 uint64 array
            nb = a.size * 8 if pytest.max_bits != -1 and pytest.max_bits <= 64 else a.size * 8 * 2
            ak.client.maxTransferBytes = nb
        else:
            a = ak.randint(0, 2 ** 32, pytest.prob_size, dtype=dtype, seed=pytest.seed)
            nb = a.size * a.itemsize
            ak.client.maxTransferBytes = nb

        npa = a.to_ndarray()
        benchmark.pedantic(ak.array, args=[npa], kwargs={"max_bits": pytest.max_bits}, rounds=pytest.trials)
        benchmark.extra_info["description"] = "Measures the performance of ak.array"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (nb / benchmark.stats["mean"]) / 2 ** 30)
        benchmark.extra_info["max_bit"] = pytest.max_bits  # useful when looking at bigint
