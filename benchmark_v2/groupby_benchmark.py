from benchmark_utils import calc_num_bytes
import pytest

import arkouda as ak

TYPES = ["int64", "bigint", "str", "mixed"]
NUM_ARR = [1, 2, 8, 16]


def generate_arrays(dtype, numArrays, N):
    arrays = []
    for i in range(numArrays):
        if dtype == ak.bigint.name:
            a = ak.randint(0, 2**32, N // numArrays, dtype=ak.uint64, seed=pytest.seed)
            b = ak.randint(0, 2**32, N // numArrays, dtype=ak.uint64, seed=pytest.seed)
            ba = ak.bigint_from_uint_arrays([a, b], max_bits=pytest.max_bits)
            arrays.append(ba)
        elif dtype == "int64" or (i % 2 == 0 and dtype == "mixed"):
            a = ak.randint(0, 2**32, N // numArrays, seed=pytest.seed)
            arrays.append(a)
        else:
            a = ak.random_strings_uniform(1, 16, N // numArrays, seed=pytest.seed)
            arrays.append(a)
        if pytest.seed is not None:
            pytest.seed += 1
    if numArrays == 1:
        arrays = arrays[0]
    return arrays


@pytest.mark.benchmark(group="GroupBy_Creation")
@pytest.mark.parametrize("numArrays", NUM_ARR)
@pytest.mark.parametrize("dtype", TYPES)
def bench_groupby(benchmark, numArrays, dtype):
    if dtype in pytest.dtype:
        N = pytest.N
        arrays = generate_arrays(dtype, numArrays, N)
        num_bytes = calc_num_bytes(arrays)

        benchmark.pedantic(ak.GroupBy, args=[arrays], rounds=pytest.trials)

        benchmark.extra_info["description"] = (
            f"Measures the performance of ak.GroupBy creation with {dtype} dtype"
        )
        benchmark.extra_info["problem_size"] = N
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
