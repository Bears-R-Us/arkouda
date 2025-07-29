import numpy as np
import pytest

import arkouda as ak

DTYPES = ("int64", "float64", "bool")


@pytest.mark.skip_numpy(True)
@pytest.mark.skip_if_rank_not_compiled(2)
@pytest.mark.benchmark(group="AK_flatten")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape_type", ["square", "wide", "tall"])
def bench_ak_flatten_2d(benchmark, dtype, shape_type):
    N = pytest.N

    if shape_type == "square":
        #   Ensure N has an integer square root:
        N = int(np.round(np.sqrt(N)) ** 2)
        sqrt_N = int(np.sqrt(N))
        shape = (sqrt_N, sqrt_N)
    elif shape_type == "tall":
        #   Ensure N is divisible by 2:
        N = int(N // 2 * 2)
        shape = (N // 2, 2)
    else:
        #   Ensure N is divisible by 2:
        N = int(N // 2 * 2)
        shape = (2, N // 2)

    if dtype == "int64":
        data = ak.randint(0, 2**32, N, dtype=ak.int64, seed=pytest.seed)
    elif dtype == "float64":
        data = ak.randint(0, 1, N, dtype=ak.float64, seed=pytest.seed)
    elif dtype == "bool":
        data = ak.randint(0, 2, N, dtype=ak.bool_, seed=pytest.seed)

    arr2d = data.reshape(shape)

    def flatten_op():
        return arr2d.flatten()

    numBytes = benchmark.pedantic(flatten_op, rounds=pytest.trials)  # noqa: F841

    benchmark.extra_info["description"] = f"Measures ak.flatten (np-style) on dtype={dtype}"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["backend"] = "Arkouda"
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (data.size * data.itemsize / benchmark.stats["mean"]) / 2**30
    )
