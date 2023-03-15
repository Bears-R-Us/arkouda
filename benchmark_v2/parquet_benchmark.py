import arkouda as ak
import pytest

import numpy as np

import tempfile
import os

tmp_dir = "{}/par_io_test".format(os.getcwd())


def multi_col_parquet_write(df):
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_dirname:
        # use multicolumn write to generate parquet file
        df.to_parquet(f"{tmp_dirname}/multicol_parquet")


def bench_multicol_parquet(benchmark):
    df_dict = {
        "c_1": ak.arange(3),
        # "c_2": ak.segarray(ak.array([0, 9, 14]), ak.arange(20)),
        "c_3": ak.arange(3, 6, dtype=ak.uint64),
        # "c_4": ak.segarray(ak.array([0, 5, 10]), ak.arange(15, dtype=ak.uint64)),
        "c_5": ak.array([False, True, False]),
        # "c_6": ak.segarray(ak.array([0, 5, 10]), ak.randint(0, 1, 15, dtype=ak.bool)),
        "c_7": ak.array(np.random.uniform(0, 100, 3)),
        # "c_8": ak.segarray(ak.array([0, 9, 14]), ak.array(np.random.uniform(0, 100, 20))),
        "c_9": ak.array(["abc", "123", "xyz"])
    }
    akdf = ak.DataFrame(df_dict)
    benchmark(multi_col_parquet_write, akdf)


def append_col_parquet_write(df):
    data = df._prep_data()
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_dirname:
        # use multicolumn write to generate parquet file
        ak.to_parquet(data, f"{tmp_dirname}/multicol_parquet", mode="append")


def bench_append_parquet(benchmark):
    benchmark.extra_info["problem_size"] = pytest.problem_size
    df_dict = {
        "c_1": ak.arange(3),
        # "c_2": ak.segarray(ak.array([0, 9, 14]), ak.arange(20)),
        "c_3": ak.arange(3, 6, dtype=ak.uint64),
        # "c_4": ak.segarray(ak.array([0, 5, 10]), ak.arange(15, dtype=ak.uint64)),
        "c_5": ak.array([False, True, False]),
        # "c_6": ak.segarray(ak.array([0, 5, 10]), ak.randint(0, 1, 15, dtype=ak.bool)),
        "c_7": ak.array(np.random.uniform(0, 100, 3)),
        # "c_8": ak.segarray(ak.array([0, 9, 14]), ak.array(np.random.uniform(0, 100, 20))),
        "c_9": ak.array(["abc", "123", "xyz"])
    }
    akdf = ak.DataFrame(df_dict)
    benchmark(append_col_parquet_write, akdf)

