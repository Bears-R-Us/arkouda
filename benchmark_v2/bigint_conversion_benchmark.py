import pytest

from benchmark_utils import calc_num_bytes

import arkouda as ak


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="BigInt_Conversion")
@pytest.mark.parametrize("direction", ["bigint_from_uint_arrays", "bigint_to_uint_arrays"])
def bench_bigint_conversion(benchmark, direction):
    N = pytest.N
    max_bits = pytest.max_bits

    a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    b = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)

    if direction == "bigint_from_uint_arrays":

        def run():
            ak.bigint_from_uint_arrays([a, b], max_bits=max_bits)

        label = "bigint_from_uint_arrays"
    else:
        ba = ak.bigint_from_uint_arrays([a, b], max_bits=max_bits)

        def run():
            ba.bigint_to_uint_arrays()

        label = "bigint_to_uint_arrays"

    benchmark.pedantic(run, rounds=pytest.trials)
    num_bytes = calc_num_bytes(a)

    benchmark.extra_info["description"] = f"Measures performance of {label} with max_bits={max_bits}"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = "Arkouda"
    benchmark.extra_info["max_bits"] = max_bits
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
