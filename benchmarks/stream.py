#!/usr/bin/env python3                                                         

import time
import numpy as np
import arkouda as ak

def time_ak_stream(N_per_locale, trials, alpha, dtype, random):
    print(">>> arkouda stream")
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
    if random:
        if dtype == 'int64':
            a = ak.randint(0, 2**32, N)
            b = ak.randint(0, 2**32, N)
        elif dtype == 'float64':
            a = ak.randint(0, 1, N, dtype=ak.float64)
            b = ak.randint(0, 1, N, dtype=ak.float64)
    else:   
        a = ak.ones(N, dtype=dtype)
        b = ak.ones(N, dtype=dtype)
    
    timings = []
    for i in range(trials):
        start = time.time()
        c = a+b*alpha
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    bytes_per_sec = (c.size * c.itemsize * 3) / tavg
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))

def time_np_stream(N, trials, alpha, dtype, random):
    print(">>> numpy stream")
    print("N = {:,}".format(N))
    if random:
        if dtype == 'int64':
            a = np.random.randint(0, 2**32, N)
            b = np.random.randint(0, 2**32, N)
        elif dtype == 'float64':
            a = np.random.random(N)
            b = np.random.random(N)
    else:
        a = np.ones(N, dtype=dtype)
        b = np.ones(N, dtype=dtype)

    timings = []
    for i in range(trials):
        start = time.time()
        c = a+b*alpha
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    bytes_per_sec = (c.size * c.itemsize * 3) / tavg
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))

def check_correctness(alpha, dtype, random):
    N = 10**4
    if random:
        if dtype == 'int64':
            a = np.random.randint(0, 2**32, N)
            b = np.random.randint(0, 2**32, N)
        elif dtype == 'float64':
            a = np.random.random(N)
            b = np.random.random(N)
    else:
        a = np.ones(N, dtype=dtype)
        b = np.ones(N, dtype=dtype)
    npc = a+b*alpha
    akc = ak.array(a)+ak.array(b)*alpha
    assert np.allclose(npc, akc.to_ndarray())

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Runs and times the stream benchmark: C = A + alpha*B in both arkouda and numpy.")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**8, help='Problem size: length of arrays A and B')
    parser.add_argument('-t', '--trials', type=int, default=6, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='int64', help='Dtype of arrays (int64 or float64)')
    parser.add_argument('-r', '--randomize', default=False, action='store_true', help='Fill arrays with random values instead of ones')
    parser.add_argument('-a', '--alpha', default=1.0, type=float, help='Scalar multiple')
    parser.add_argument('-c', '--check', default=False, action='store_true', help='Run on a small array first to test whether arkouda and numpy results agree')
    args = parser.parse_args()
    if args.dtype not in ('int64', 'float64'):
        raise ValueError("Dtype must be either int64 or float64, not {}".format(args.dtype))
    ak.v = False
    ak.connect(args.hostname, args.port)
    if args.check:
        print("Verifying correctness on small problem... ", end="")
        check_correctness(args.alpha, args.dtype, args.randomize)
        print("CORRECT")
    
    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_stream(args.size, args.trials, args.alpha, args.dtype, args.randomize)
    time_np_stream(args.size, args.trials, args.alpha, args.dtype, args.randomize)
    sys.exit(0)
