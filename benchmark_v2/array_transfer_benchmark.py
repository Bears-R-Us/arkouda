import gc

import pytest

import arkouda as ak


TYPES = ("int64", "float64", "bigint")


def create_ak_array(N, dtype, max_bits=pytest.max_bits):
    if dtype == ak.bigint.name:
        u1 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
        u2 = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=pytest.seed)
        a = ak.bigint_from_uint_arrays([u1, u2], max_bits=max_bits)
        num_bytes = a.size * 8 if max_bits != -1 and max_bits <= 64 else a.size * 8 * 2
    else:
        a = ak.randint(0, 2**32, N, dtype=dtype, seed=pytest.seed)
        num_bytes = a.size * a.itemsize
    return a, num_bytes


@pytest.mark.benchmark(group="ArrayTransfer_tondarray")
@pytest.mark.parametrize("dtype", TYPES)
def bench_array_transfer_to_ndarray(benchmark, dtype):
    if dtype in pytest.dtype:
        n = pytest.prob_size  # use the per-locale problem size, not N
        a, num_bytes = create_ak_array(n, dtype)
        ak.core.client.maxTransferBytes = num_bytes

        def setup():
            gc.collect()
            return (), {}

        def teardown():
            gc.collect()

        def to_nd():
            a.to_ndarray()

        benchmark.pedantic(
            to_nd,
            setup=setup,
            teardown=teardown,
            rounds=pytest.trials,
        )

        benchmark.extra_info["description"] = "Measures the performance of pdarray.to_ndarray"
        benchmark.extra_info["problem_size"] = n
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
        benchmark.extra_info["max_bit"] = pytest.max_bits


@pytest.mark.benchmark(group="ArrayTransfer_ak.array")
@pytest.mark.parametrize("dtype", TYPES)
def bench_array_transfer_from_ndarray(benchmark, dtype):
    if dtype in pytest.dtype:
        n = pytest.prob_size  # use the per-locale problem size, not N
        a, num_bytes = create_ak_array(n, dtype)
        ak.core.client.maxTransferBytes = num_bytes
        npa = a.to_ndarray()

        def setup():
            gc.collect()
            return (), {}

        def teardown():
            gc.collect()

        def from_np():
            ak.array(npa, max_bits=pytest.max_bits)

        def from_np_bigint():
            ak.array(npa, max_bits=-1, dtype=dtype, unsafe=True, num_bits=128, any_neg=False)

        if dtype == "bigint":
            benchmark.pedantic(
                from_np_bigint,
                setup=setup,
                teardown=teardown,
                rounds=pytest.trials,
            )
        else:
            benchmark.pedantic(
                from_np,
                setup=setup,
                teardown=teardown,
                rounds=pytest.trials,
            )

        benchmark.extra_info["description"] = "Measures the performance of ak.array"
        benchmark.extra_info["problem_size"] = n
        benchmark.extra_info["num_bytes"] = num_bytes
        #   units are GiB/sec:
        benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
        benchmark.extra_info["max_bit"] = pytest.max_bits
