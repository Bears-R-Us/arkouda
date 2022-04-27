#!/usr/bin/env python3

import time, argparse
import arkouda as ak
import numpy as np
import os
import tempfile

def make_data(N_per_locale, dir, seed):
    def increment(x):
        if x is None:
            return x
        else:
            return x+1
    cfg = ak.get_config()
    N = (N_per_locale * cfg["numLocales"]) // 5
    maxval = int(np.sqrt(N/26))
    with tempfile.TemporaryDirectory(dir=dir) as tmp_dirname:
        paths = []
        # Write data as 5 separate datasets
        for dset in range(5):
            df = ak.DataFrame()
            # Generate data
            for dt in (ak.int64, ak.uint64, ak.float64, ak.bool):
                df[dt.name] = ak.cast(ak.randint(0, maxval, N, seed=seed), dt)
                seed = increment(seed)
            df['str'] = ak.random_strings_uniform(minlen=1, maxlen=2, size=N, seed=seed)
            seed = increment(seed)
            df['cat'] = ak.Categorical(df['str'])
            df['vals'] = ak.randint(0, maxval, N, seed=seed)
            seed = increment(seed)
            # Write and read
            path = os.path.join(tmp_dirname, 'test_data_{}'.format(dset))
            df.save(path)
            paths.append(path)

        df = ak.DataFrame()
        # Read datasets back in one at a time to test append
        for path in paths:
            df = df.append(ak.DataFrame(ak.read(path+'*')))
        assert (df.size == 5*N), "Wrong size: expected {:,}, got {:,}".format(5*N, df.size)
    return df

def aggregate(df):
    key = ('int64', 'uint64', 'bool', 'str', 'cat')
    g = df.groupby(key)
    agg = ak.DataFrame()
    for k, x in zip(key, agg.unique_keys):
        agg[k] = x
    for op in ('sum', 'mean', 'min', 'max', 'OR', 'AND', 'nunique'):
        agg[op] = g.aggregate(df['vals'], op)
    return agg

def enrich(df):
    keycols = ('int64', 'uint64', 'str', 'cat')
    table = ak.DataFrame()
    keys = ak.unique([df[k] for k in keycols])
    vals = ak.arange(keys[0].size)
    args = [df[k] for k in keycols]
    return ak.lookup(keys, vals, args)
    
def filter(df):
    perm = ak.argsort(df['float64'])
    vals = df['float64'][perm]
    lower_quantile = vals[int(0.05*df.size)]
    upper_quantile = vals[int(0.95*df.size)]
    f = (df['float64'] > lower_quantile) & (df['float64'] < upper_quantile)
    return f

def main(N_per_locale, dir, seed):
    t = time.time()
    df = make_data(N_per_locale, dir, seed)
    e = time.time() - t
    print("Generate/write/read time = {:.3f}".format(e))

    t = time.time()
    df = df[filter(df)]
    e = time.time() - t
    print("Filter time = {:.3f}".format(e))

    t = time.time()
    agg = aggregate(df)
    e = time.time() - t
    print("Aggregate time = {:.3f}".format(e))

    t = time.time()
    agg['enrich'] = enrich(agg)
    e = time.time() - t
    print("Enrich time = {:.3f}".format(e))
    
    agg.register('agg_')
    return agg

def create_parser():
    parser = argparse.ArgumentParser(description="Measure runtime of a DataFrame-based workflow.")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**7, help='Problem size: number of rows in the DataFrame')
    parser.add_argument('-p', '--path', default=os.getcwd(), help='Target path for writing/reading temporary data')
    parser.add_argument('--correctness-only', default=False, action='store_true', help='Only check correctness, not performance.')
    parser.add_argument('-s', '--seed', default=None, type=int, help='Value to initialize random number generator')
    # parser.add_argument('-q', '--parquet', default=False, action='store_true', help='Perform Parquet operations')
    return parser
    
if __name__ == "__main__":
    import sys
    parser = create_parser()
    args = parser.parse_args()
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        main(1_000, args.path, None)
        sys.exit(0)
    
    print("DataFrame rows per locale = {:,}".format(args.size))
    print("number of trials = ", args.trials)

    main(args.size, args.path, args.seed)
    
    sys.exit(0)
