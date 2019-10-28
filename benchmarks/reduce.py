#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak

OPS = ('sum', 'prod', 'min', 'max')

def time_ak_reduce(N_per_locale, trials, dtype, random):
    print(">>> arkouda reduce")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
    if random:
        if dtype == 'int64':
            a = ak.randint(0, 2**32, N)
        elif dtype == 'float64':
            a = ak.randint(0, 1, N, dtype=ak.float64)
    else:
        a = ak.arange(0, N, 1)
        if dtype == 'float64':
            a = 1.0 * a
     
    timings = {op: [] for op in OPS}
    results = {}
    for i in range(trials):
        for op in timings.keys():
            fxn = getattr(a, op)
            start = time.time()
            r = fxn()
            end = time.time()
            timings[op].append(end - start)
            results[op] = r
    tavg = {op: sum(t) / trials for op, t in timings.items()}

    for op, t in tavg.items():
        print("{} = {}".format(op, results[op]))
        print("  Average time = {:.4f} sec".format(t))
        bytes_per_sec = (a.size * a.itemsize) / t
        print("  Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))

def time_np_reduce(N, trials, dtype, random):
    print(">>> numpy reduce")
    print("N = {:,}".format(N))
    if random:
        if dtype == 'int64':
            a = np.random.randint(0, 2**32, N)
        elif dtype == 'float64':
            a = np.random.random(N)
    else:   
        a = np.arange(0, N, 1, dtype=dtype)
     
    timings = {op: [] for op in OPS}
    results = {}
    for i in range(trials):
        for op in timings.keys():
            fxn = getattr(a, op)
            start = time.time()
            r = fxn()
            end = time.time()
            timings[op].append(end - start)
            results[op] = r
    tavg = {op: sum(t) / trials for op, t in timings.items()}

    for op, t in tavg.items():
        print("{} = {}".format(op, results[op]))
        print("  Average time = {:.4f} sec".format(t))
        bytes_per_sec = (a.size * a.itemsize) / t
        print("  Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))
    
def create_parser():
    parser = argparse.ArgumentParser(description="Measure performance of reductions over arrays.")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**8, help='Problem size: length of array to reduce')
    parser.add_argument('-t', '--trials', type=int, default=6, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='int64', help='Dtype of array (int64 or float64)')
    parser.add_argument('-r', '--randomize', default=False, action='store_true', help='Fill array with random values instead of range')
    parser.add_argument('--numpy', default=False, action='store_true', help='Run the same operation in NumPy to compare performance.')
    return parser
        
if __name__ == "__main__":
    import sys
    parser = create_parser()
    args = parser.parse_args()
    if args.dtype not in ('int64', 'float64'):
        raise ValueError("Dtype must be either int64 or float64, not {}".format(args.dtype))
    ak.verbose = False
    ak.connect(args.hostname, args.port)
    
    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_reduce(args.size, args.trials, args.dtype, args.randomize)
    if args.numpy:
        time_np_reduce(args.size, args.trials, args.dtype, args.randomize)
    sys.exit(0)
