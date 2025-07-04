import pytest

import arkouda as ak

TYPES = ["int64", "bigint", "str", "mixed"]
NUM_ARR = [1, 2, 8, 16]


def generate_arrays(dtype, numArrays, N):
    totalbytes = 0
    arrays = []
    for i in range(numArrays):
        if dtype == ak.bigint.name:
            a = ak.randint(0, 2**32, N // numArrays, dtype=ak.uint64, seed=pytest.seed)
            b = ak.randint(0, 2**32, N // numArrays, dtype=ak.uint64, seed=pytest.seed)
            ba = ak.bigint_from_uint_arrays([a, b], max_bits=pytest.max_bits)
            arrays.append(ba)
            totalbytes += a.size * 8 if pytest.max_bits != -1 and pytest.max_bits <= 64 else a.size * 16
        elif dtype == "int64" or (i % 2 == 0 and dtype == "mixed"):
            a = ak.randint(0, 2**32, N // numArrays, seed=pytest.seed)
            arrays.append(a)
            totalbytes += a.size * a.itemsize
        else:
            a = ak.random_strings_uniform(1, 16, N // numArrays, seed=pytest.seed)
            arrays.append(a)
            totalbytes += a.nbytes * a.entry.itemsize
        if pytest.seed is not None:
            pytest.seed += 1
    if numArrays == 1:
        arrays = arrays[0]
    return arrays, totalbytes


@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="GroupBy_Creation")
@pytest.mark.parametrize("numArrays", NUM_ARR)
@pytest.mark.parametrize("dtype", TYPES)
def bench_groupby(benchmark, numArrays, dtype):
    if dtype in pytest.dtype:
        N = 10**4 if pytest.correctness_only else pytest.prob_size * ak.get_config()["numLocales"]
        arrays, totalbytes = generate_arrays(dtype, numArrays, N)

        def create_groupby():
            _ = ak.GroupBy(arrays)
            return totalbytes

        numBytes = benchmark.pedantic(create_groupby, rounds=pytest.trials)

        benchmark.extra_info["description"] = (
            f"Measures the performance of ak.GroupBy creation with {dtype} dtype"
        )
        benchmark.extra_info["problem_size"] = N
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (numBytes / benchmark.stats["mean"]) / 2**30
        )
