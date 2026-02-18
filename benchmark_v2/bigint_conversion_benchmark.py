import pytest

import arkouda as ak


@pytest.mark.skip_numpy(True)
@pytest.mark.benchmark(group="BigInt_Conversion")
@pytest.mark.parametrize("direction", ["to_bigint", "from_bigint"])
def bench_bigint_conversion(benchmark, direction):
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]
    max_bits = pytest.max_bits

    a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    b = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
    tot_bytes = N * 8 if (0 < max_bits <= 64) else N * 16

    if direction == "to_bigint":

        def run():
            ak.bigint_from_uint_arrays([a, b], max_bits=max_bits)
            return tot_bytes

        label = "bigint_from_uint_arrays"
    else:
        ba = ak.bigint_from_uint_arrays([a, b], max_bits=max_bits)

        def run():
            ba.bigint_to_uint_arrays()
            return tot_bytes

        label = "bigint_to_uint_arrays"

    bytes_processed = benchmark.pedantic(run, rounds=pytest.trials)

    benchmark.extra_info["description"] = f"Measures performance of {label} with max_bits={max_bits}"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = "Arkouda"
    benchmark.extra_info["max_bits"] = max_bits
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (bytes_processed / benchmark.stats["mean"]) / 2**30
    )
