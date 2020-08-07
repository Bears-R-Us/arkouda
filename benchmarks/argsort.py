#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak

TYPES = ('int64', 'float64')

def time_ak_argsort(N_per_locale, trials, dtype):
    print(">>> arkouda argsort")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
    if dtype == 'int64':
        a = ak.randint(0, 2**32, N)
    elif dtype == 'float64':
        a = ak.randint(0, 1, N, dtype=ak.float64)
     
    timings = []
    for i in range(trials):
        start = time.time()
        perm = ak.argsort(a)
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    assert ak.is_sorted(a[perm])
    print("Average time = {:.4f} sec".format(tavg))
    bytes_per_sec = (a.size * a.itemsize) / tavg
    print("Average rate = {:.4f} GiB/sec".format(bytes_per_sec/2**30))

def time_np_argsort(N, trials, dtype):
    print(">>> numpy argsort")
    print("N = {:,}".format(N))
    if dtype == 'int64':
        a = np.random.randint(0, 2**32, N)
    elif dtype == 'float64':
        a = np.random.random(N)
     
    timings = []
    for i in range(trials):
        start = time.time()
        perm = np.argsort(a)
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    bytes_per_sec = (a.size * a.itemsize) / tavg
    print("Average rate = {:.4f} GiB/sec".format(bytes_per_sec/2**30))

def check_correctness(dtype):
    N = 10**4
    if dtype == 'int64':
        a = ak.randint(0, 2**32, N)
    elif dtype == 'float64':
        a = ak.randint(0, 1, N, dtype=ak.float64)

    perm = ak.argsort(a)
    assert ak.is_sorted(a[perm])

def create_parser():
    parser = argparse.ArgumentParser(description="Measure performance of sorting an array of random values.")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**8, help='Problem size: length of array to argsort')
    parser.add_argument('-t', '--trials', type=int, default=3, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='int64', help='Dtype of array ({})'.format(', '.join(TYPES)))
    parser.add_argument('--numpy', default=False, action='store_true', help='Run the same operation in NumPy to compare performance.')
    parser.add_argument('--correctness-only', default=False, action='store_true', help='Only check correctness, not performance.')
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
            check_correctness(dtype)
        sys.exit(0)
    
    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_argsort(args.size, args.trials, args.dtype)
    if args.numpy:
        time_np_argsort(args.size, args.trials, args.dtype)
    sys.exit(0)
