#!/usr/bin/env python3                                                         

import time, argparse
import arkouda as ak
import os
from glob import glob

TYPES = ('int64')

def time_ak_1d(N_per_locale, trials, dtype, path, seed):
    print(">>> arkouda {} 1D Binary Operation".format(dtype))
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
    a = ak.randint(0, 2**32, N, seed=seed)
    b = ak.randint(0, 2**32, N, seed=seed)
     
    writetimes = []
    readtimes = []
    for i in range(trials):
        start = time.time()
        c = a + b
        end = time.time()
        readtimes.append(end - start)
        for f in glob(path+'_LOCALE*'):
            os.remove(f)
    avgread = sum(readtimes) / trials

    print("1D Average time = {:.4f} sec".format(avgread))

    nb = a.size * a.itemsize
    print("1D Average rate = {:.2f} GiB/sec".format(nb/2**30/avgread))

def time_ak_2d(N_per_locale, M_per_locale, trials, dtype, path, seed):
    print(">>> arkouda {} 2D Binary Operation".format(dtype))
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    M = M_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
    a = ak.randint2D(0, 2**32, M, N, seed=seed)
    b = ak.randint2D(0, 2**32, M, N, seed=seed)
     
    writetimes = []
    readtimes = []
    for i in range(trials):
        start = time.time()
        c = a + b
        end = time.time()
        readtimes.append(end - start)
        for f in glob(path+'_LOCALE*'):
            os.remove(f)
    avgread = sum(readtimes) / trials

    print("2D Average time = {:.4f} sec".format(avgread))

    nb = a.size * a.itemsize
    print("2D Average rate = {:.2f} GiB/sec".format(nb/2**30/avgread))

    
def check_correctness(dtype, path, seed):
    N = 10**4
    if dtype == 'int64':
        a = ak.randint(0, 2**32, N, seed=seed)
    elif dtype == 'float64':
        a = ak.randint(0, 1, N, dtype=ak.float64, seed=seed)

    a.save(path)
    b = ak.load(path)
    for f in glob(path+"_LOCALE*"):
        os.remove(f)
    assert (a == b).all()

def create_parser():
    parser = argparse.ArgumentParser(description="Measure performance of writing and reading a random array from disk.")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**2, help='Problem size: length of array to write/read')
    parser.add_argument('-m', '--size2', type=int, default=10**2, help='Problem size: length of array to write/read')  
    parser.add_argument('-t', '--trials', type=int, default=1, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='int64', help='Dtype of array ({})'.format(', '.join(TYPES)))
    parser.add_argument('-p', '--path', default=os.getcwd()+'ak-io-test', help='Target path for measuring read/write rates')
    parser.add_argument('--correctness-only', default=False, action='store_true', help='Only check correctness, not performance.')
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
            check_correctness(dtype, args.path, args.seed)
        sys.exit(0)
    
    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_1d(args.size*args.size2, args.trials, args.dtype, args.path, args.seed)
    time_ak_2d(args.size, args.size2, args.trials, args.dtype, args.path, args.seed)
    sys.exit(0)
