import arkouda as ak
import pytest

# mixed used for groupby of str and int64
TYPES = ["int64", "bigint", "str", "mixed"]
NUM_ARR = [1, 2, 8, 16]


def generate_arrays(dtype, numArrays):
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]

    totalbytes = 0
    arrays = []
    for i in range(numArrays):
        if dtype == ak.bigint.name:
            a = ak.randint(0, 2**32, N // numArrays, dtype=ak.uint64, seed=pytest.seed)
            b = ak.randint(0, 2**32, N // numArrays, dtype=ak.uint64, seed=pytest.seed)
            ba = ak.bigint_from_uint_arrays([a, b], max_bits=pytest.max_bits)
            arrays.append(ba)
            # bytes per bigint array (N * 16) since it's made of 2 uint64 arrays
            # if max_bits in [0, 64] then they're essentially 1 uint64 array
            totalbytes += (
                a.size * 8 if pytest.max_bits != -1 and pytest.max_bits <= 64 else a.size * 8 * 2
            )
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
        arrays, totalbytes = generate_arrays(dtype, numArrays)
        benchmark.pedantic(ak.GroupBy, args=[arrays], rounds=pytest.trials)

        benchmark.extra_info["description"] = "Measures the performance of ak.GroupBy"
        benchmark.extra_info["problem_size"] = pytest.prob_size
        benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
            (totalbytes / benchmark.stats["mean"]) / 2 ** 30)
