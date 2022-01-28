import time, argparse
import arkouda as ak
import os
from glob import glob
import math

TYPES = ('int64')

def time_binary_op(N_per_locale, trials, dtype, seed):
    print(">>> arkouda {} binary op".format(dtype))
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
    a = ak.randint(0, 2**32, N)
    b = ak.randint2D(0,2**32, int(math.sqrt(N)), int(math.sqrt(N)))
     
    times1d = []
    times2d = []
    for i in range(trials):
        start = time.time()
        a + a
        end = time.time()
        times1d.append(end - start)
        start = time.time()
        b + b
        end = time.time()
        times2d.append(end - start)
    avg1d = sum(times1d) / trials
    avg2d = sum(times2d) / trials

    print("1d Average time = {:.4f} sec".format(avg1d))
    print("2d Average time = {:.4f} sec".format(avg2d))

    nb = a.size * a.itemsize
    print("1d Average rate = {:.2f} GiB/sec".format(nb/2**30/avg1d))
    print("2d Average rate = {:.2f} GiB/sec".format(nb/2**30/avg2d))

def check_correctness(dtype, path, seed):
    N = 10**4
    a = ak.randint(0, 2**32, N, seed=seed)

    a.save_parquet(path)
    b = ak.read_parquet(path+'*')
    for f in glob(path + '_LOCALE*'):
        os.remove(f)
    assert (a == b).all()

def create_parser():
    parser = argparse.ArgumentParser(description="Measure performance of writing and reading a random array from disk.")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**8, help='Problem size: length of array to write/read')
    parser.add_argument('-t', '--trials', type=int, default=1, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='int64', help='Dtype of array ({})'.format(', '.join(TYPES)))
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

    #if args.correctness_only:
    #   for dtype in TYPES:
    #      check_correctness(dtype, args.seed)
    # sys.exit(0)
    
    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_binary_op(args.size, args.trials, args.dtype, args.seed)
    sys.exit(0)
