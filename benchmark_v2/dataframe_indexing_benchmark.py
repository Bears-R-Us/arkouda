#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pytest

import arkouda as ak

OPS = ["_get_head_tail_server", "_get_head_tail"]


def generate_dataframe():
    cfg = ak.get_config()
    N = pytest.prob_size * cfg["numLocales"]
    types = [ak.Categorical, ak.pdarray, ak.Strings, ak.SegArray]

    # generate random columns to build dataframe
    df_dict = {}
    np.random.seed(pytest.seed)
    for x in range(20):  # loop to create 20 random columns
        key = f"c_{x}"
        d = types[x % 4]
        if d == ak.Categorical:
            str_arr = ak.random_strings_uniform(minlen=5, maxlen=6, size=N, seed=pytest.seed)
            df_dict[key] = ak.Categorical(str_arr)
        elif d == ak.pdarray:
            df_dict[key] = ak.array(np.random.randint(0, 2**32, N))
        elif d == ak.Strings:
            df_dict[key] = ak.random_strings_uniform(minlen=5, maxlen=6, size=N, seed=pytest.seed)
        elif d == ak.SegArray:
            df_dict[key] = ak.SegArray(ak.arange(0, N), ak.array(np.random.randint(0, 2**32, N)))
    return ak.DataFrame(df_dict)


@pytest.mark.skip_correctness_only(True)
@pytest.mark.benchmark(group="Dataframe_Indexing")
@pytest.mark.parametrize("op", OPS)
def bench_dataframe(benchmark, op):
    """
    Measures the performance of arkouda Dataframe indexing

    """
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.min_rows", 10)
    pd.set_option("display.max_columns", 20)

    df = generate_dataframe()

    fxn = getattr(df, op)
    benchmark.pedantic(fxn, rounds=pytest.trials)

    # calculate nbytes based on the columns
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
            nbytes += col_obj.values.size * col_obj.values.itemsize + (
                col_obj.segments.size * col_obj.segments.itemsize
            )

    benchmark.extra_info["description"] = "Measures the performance of arkouda Dataframe indexing"
    benchmark.extra_info["problem_size"] = pytest.prob_size
    benchmark.extra_info["transfer_rate"] = "{:.4f} GiB/sec".format(
        (nbytes / benchmark.stats["mean"]) / 2**30
    )
