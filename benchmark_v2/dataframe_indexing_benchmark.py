#!/usr/bin/env python3
import numpy as np
import pytest

from benchmark_utils import calc_num_bytes

import arkouda as ak


OPS = ["_get_head_tail_server", "_get_head_tail"]


@pytest.mark.benchmark(group="Dataframe_Indexing")
@pytest.mark.parametrize("op", OPS)
def bench_dataframe(benchmark, op):
    """Measures the performance of arkouda Dataframe indexing."""
    N = pytest.N

    types = [ak.Categorical, ak.pdarray, ak.Strings, ak.SegArray]
    df_dict = {}
    np.random.seed(pytest.seed)

    for x in range(20):
        key = f"c_{x}"
        dtype = types[x % 4]
        if dtype == ak.Categorical:
            str_arr = ak.random_strings_uniform(5, 6, N, seed=pytest.seed)
            df_dict[key] = ak.Categorical(str_arr)
        elif dtype == ak.pdarray:
            df_dict[key] = ak.array(np.random.randint(0, 2**32, N))
        elif dtype == ak.Strings:
            df_dict[key] = ak.random_strings_uniform(5, 6, N, seed=pytest.seed)
        elif dtype == ak.SegArray:
            df_dict[key] = ak.SegArray(ak.arange(0, N), ak.array(np.random.randint(0, 2**32, N)))

    df = ak.DataFrame(df_dict)

    # calculate nbytes
    num_bytes = calc_num_bytes(df)

    if pytest.numpy:
        # simulate with pandas
        df_sim = df.to_pandas()

        def pandas_op():
            if op == "_get_head_tail_server" or op == "_get_head_tail":
                df_sim.head()  # simplified equivalent

        benchmark.pedantic(pandas_op, rounds=pytest.trials)
    else:
        fxn = getattr(df, op)

        def arkouda_op():
            fxn()

        benchmark.pedantic(arkouda_op, rounds=pytest.trials)

    benchmark.extra_info["description"] = "Measures the performance of arkouda Dataframe indexing"
    benchmark.extra_info["problem_size"] = N
    benchmark.extra_info["num_bytes"] = num_bytes
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((num_bytes / benchmark.stats["mean"]) / 2**30)
