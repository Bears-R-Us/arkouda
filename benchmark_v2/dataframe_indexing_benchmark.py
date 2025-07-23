#!/usr/bin/env python3
import numpy as np
import pytest

import arkouda as ak

OPS = ["_get_head_tail_server", "_get_head_tail"]


@pytest.mark.benchmark(group="Dataframe_Indexing")
@pytest.mark.parametrize("op", OPS)
def bench_dataframe(benchmark, op):
    """
    Measures the performance of arkouda Dataframe indexing
    """
    N = pytest.prob_size * ak.get_config()["numLocales"]

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
    nbytes = 0
    for col in df.columns:
        col_obj = df[col]
        if isinstance(col_obj, ak.pdarray):
            nbytes += col_obj.size * col_obj.itemsize
        elif isinstance(col_obj, ak.Categorical):
            nbytes += col_obj.codes.size * col_obj.codes.itemsize
        elif isinstance(col_obj, ak.Strings):
            nbytes += col_obj.nbytes * col_obj.entry.itemsize
        elif isinstance(col_obj, ak.SegArray):
            nbytes += (
                col_obj.values.size * col_obj.values.itemsize
                + col_obj.segments.size * col_obj.segments.itemsize
            )

    if pytest.numpy:
        # simulate with pandas
        df_sim = df.to_pandas()

        def pandas_op():
            if op == "_get_head_tail_server" or op == "_get_head_tail":
                df_sim.head()  # simplified equivalent
            return nbytes

        numBytes = benchmark.pedantic(pandas_op, rounds=pytest.trials)
    else:
        fxn = getattr(df, op)

        def arkouda_op():
            fxn()
            return nbytes

        numBytes = benchmark.pedantic(arkouda_op, rounds=pytest.trials)

    benchmark.extra_info["description"] = "Measures the performance of arkouda Dataframe indexing"
    benchmark.extra_info["problem_size"] = N
    #   units are GiB/sec:
    benchmark.extra_info["transfer_rate"] = float((numBytes / benchmark.stats["mean"]) / 2**30)
