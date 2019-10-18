#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak

def time_ak_argsort(N_per_locale, trials, dtype, scale_by_locales):
    print(">>> arkouda argsort")
    cfg = ak.get_config()
    if scale_by_locales:
        N = N_per_locale * cfg["numLocales"]
    else:
        N = N_per_locale
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

def create_parser():
    parser = argparse.ArgumentParser(description="Measure performance of sorting an array of random values.")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**7, help='Problem size: length of array to argsort')
    parser.add_argument('-t', '--trials', type=int, default=6, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='int64', help='Dtype of array (int64 or float64)')
    parser.add_argument('-s', '--scale-by-locales', default=False, action='store_true', help='For arkouda, scale up the array by the number of locales')
    parser.add_argument('--numpy', default=False, action='store_true', help='Run the same operation in NumPy to compare performance.')
    return parser

if __name__ == "__main__":
    import sys
    parser = create_parser()
    args = parser.parse_args()
    if args.dtype not in ('int64', 'float64'):
        raise ValueError("Dtype must be either int64 or float64, not {}".format(args.dtype))
    ak.v = False
    ak.connect(args.hostname, args.port)
    
    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_argsort(args.size, args.trials, args.dtype, args.scale_by_locales)
    if args.numpy:
        time_np_argsort(args.size, args.trials, args.dtype)
    sys.exit(0)
