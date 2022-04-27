#!/usr/bin/env python3
import string
import time, argparse
import numpy as np
import pandas as pd

import arkouda as ak
import random

OPS = ['_repr_html_', '_get_head_tail_server', '_get_head_tail']
TYPES = ('int64', 'uint64',)

def generate_dataframe(N):
    types = [ak.Categorical, ak.pdarray, ak.Strings, ak.SegArray]

    # generate random columns to build dataframe
    df_dict = {}
    for x in range(20):  # loop to create 20 random columns
        key = f"c_{x}"
        d = types[random.randint(0, len(types)-1)]
        if d == ak.Categorical:
            str_arr = ak.array(["".join(random.choices(string.ascii_letters + string.digits, k=5)) for _ in range(N)])
            df_dict[key] = ak.Categorical(str_arr)
        elif d == ak.pdarray:
            df_dict[key] = ak.array(np.random.randint(0, 2 ** 32, N))
        elif d == ak.Strings:
            df_dict[key] = ak.array(["".join(random.choices(string.ascii_letters + string.digits, k=5)) for _ in range(N)])
        elif d == ak.SegArray:
            df_dict[key] = ak.SegArray(ak.arange(0, N*5, 5), ak.array(np.random.randint(0, 2 ** 32, N*5)))

    return ak.DataFrame(df_dict)

def time_ak_df_display(N_per_locale, trials):
    print(">>> arkouda dataframe display")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.min_rows", 10)
    pd.set_option("display.max_columns", 20)

    df = generate_dataframe(N)

    timings = {op: [] for op in OPS}
    results = {}
    for i in range(trials):
        timings = {op: [] for op in OPS}
        for op in timings.keys():
            fxn = getattr(df, op)
            start = time.time()
            r = fxn()
            end = time.time()
            timings[op].append(end - start)
            results[op] = r

    tavg = {op: sum(t) / trials for op, t in timings.items()}

    for op, t in tavg.items():
        print("  {} Average time = {:.4f} sec".format(op, t))
        bytes_per_sec = (df.size * 64 * 2) / t
        print("  {} Average rate = {:.2f} GiB/sec".format(op, bytes_per_sec / 2 ** 30))

def check_correctness(N_per_locale):
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    df = generate_dataframe(N)

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.min_rows", 10)
    pd.set_option("display.max_columns", 20)

    printdf = df._get_head_tail_server() # measure the pandas df returned
    # Mainly want to verify shape for the print
    assert(printdf.shape[0] == 101)
    assert(printdf.shape[1] == 20)


def create_parser():
    parser = argparse.ArgumentParser(description="Run the setops benchmarks: intersect1d, union1d, setdiff1d, setxor1d")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**4, help='Problem size: length of arrays A and B')
    parser.add_argument('-t', '--trials', type=int, default=1, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='int64', help='Dtype of array ({})'.format(', '.join(TYPES)))
    parser.add_argument('--correctness-only', default=False, action='store_true',
                        help='Only check correctness, not performance.')
    parser.add_argument('-s', '--seed', default=None, type=int, help='Value to initialize random number generator')
    return parser


if __name__ == "__main__":
    import sys

    parser = create_parser()
    args = parser.parse_args()
    if args.dtype not in TYPES:
        raise ValueError("Dtype must be {}, not {}".format('/'.join(TYPES), args.dtype))

    ak.verbose = False
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        for dtype in TYPES:
            check_correctness(args.size)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_df_display(args.size, args.trials)

    sys.exit(0)
