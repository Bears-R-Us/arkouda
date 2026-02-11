#!/usr/bin/env python3

import argparse
import time

import numpy as np
import pandas as pd

import arkouda as ak


OPS = ["_get_head_tail_server", "_get_head_tail"]
TYPES = (
    "int64",
    "uint64",
)


def generate_dataframe(N, seed):
    types = [ak.Categorical, ak.pdarray, ak.Strings, ak.SegArray]

    # generate random columns to build dataframe
    df_dict = {}
    np.random.seed(seed)
    for x in range(20):  # loop to create 20 random columns
        key = f"c_{x}"
        d = types[x % 4]
        if d == ak.Categorical:
            str_arr = ak.random_strings_uniform(minlen=5, maxlen=6, size=N, seed=seed)
            df_dict[key] = ak.Categorical(str_arr)
        elif d == ak.pdarray:
            df_dict[key] = ak.array(np.random.randint(0, 2**32, N))
        elif d == ak.Strings:
            df_dict[key] = ak.random_strings_uniform(minlen=5, maxlen=6, size=N, seed=seed)
        elif d == ak.SegArray:
            df_dict[key] = ak.SegArray(ak.arange(0, N), ak.array(np.random.randint(0, 2**32, N)))
    return ak.DataFrame(df_dict)


def time_ak_df_display(N_per_locale, trials, seed):
    print(">>> arkouda dataframe display")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numNodes"]
    print("numLocales = {}, numNodes {}, N = {:,}".format(cfg["numLocales"], cfg["numNodes"], N))

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.min_rows", 10)
    pd.set_option("display.max_columns", 20)

    df = generate_dataframe(N, seed)

    timings = {op: [] for op in OPS}
    results = {}
    for i in range(trials):
        for op in timings.keys():
            fxn = getattr(df, op)
            start = time.time()
            r = fxn()
            end = time.time()
            timings[op].append(end - start)
            results[op] = r

    tavg = {op: sum(t) / trials for op, t in timings.items()}

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

    for op, t in tavg.items():
        print("  {} Average time = {:.4f} sec".format(op, t))
        bytes_per_sec = nbytes / t
        print("  {} Average rate = {:.2f} GiB/sec".format(op, bytes_per_sec / 2**30))


def check_correctness(N_per_locale, seed):
    cfg = ak.get_config()
    N = N_per_locale * cfg["numNodes"]
    df = generate_dataframe(N, seed)

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.min_rows", 10)
    pd.set_option("display.max_columns", 20)

    printdf = df._get_head_tail_server()  # measure the pandas df returned
    # Mainly want to verify shape for the print
    assert printdf.shape[0] == 101
    assert printdf.shape[1] == 20


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run the dataframe display benchmarks: _get_head_tail, _get_head_tail_server"
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n", "--size", type=int, default=10**4, help="Problem size: length of columns in dataframe."
    )
    parser.add_argument(
        "-t", "--trials", type=int, default=1, help="Number of times to run the benchmark"
    )
    parser.add_argument(
        "-d", "--dtype", default="int64", help="Dtype of array ({})".format(", ".join(TYPES))
    )
    parser.add_argument(
        "--correctness-only",
        default=False,
        action="store_true",
        help="Only check correctness, not performance.",
    )
    parser.add_argument(
        "-s", "--seed", default=None, type=int, help="Value to initialize random number generator"
    )
    return parser


if __name__ == "__main__":
    import sys

    parser = create_parser()
    args = parser.parse_args()
    if args.dtype not in TYPES:
        raise ValueError("Dtype must be {}, not {}".format("/".join(TYPES), args.dtype))

    ak.verbose = False
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        check_correctness(args.size, args.seed)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_df_display(args.size, args.trials, args.seed)
    sys.exit(0)
