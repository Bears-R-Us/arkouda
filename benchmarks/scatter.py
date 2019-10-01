#!/usr/bin/env python3                                                         

import time
import numpy as np
import arkouda as ak

def time_ak_scatter(isize, vsize, trials, dtype, random):
    print(">>> arkouda scatter")
    cfg = ak.get_config()
    Ni = isize * cfg["numLocales"]
    Nv = vsize * cfg["numLocales"]
    print("numLocales = {}, num_indices = {:,} ; num_values = {:,}".format(cfg["numLocales"], Ni, Nv))
    # Index vector is always random
    i = ak.randint(0, Nv, Ni)
    c = ak.zeros(Nv, dtype=dtype)
    if random:
        if dtype == 'int64':
            v = ak.randint(0, 2**32, Ni)
        elif dtype == 'float64':
            v = ak.randint(0, 1, Ni, dtype=ak.float64)
    else:   
        v = ak.ones(Ni, dtype=dtype)
    
    timings = []
    for _ in range(trials):
        start = time.time()
        c[i] = v
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    bytes_per_sec = (i.size * i.itemsize * 3) / tavg
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))

def time_np_scatter(Ni, Nv, trials, dtype, random):
    print(">>> numpy scatter")
    print("num_indices = {:,} ; num_values = {:,}".format(Ni, Nv))
    # Index vector is always random
    i = np.random.randint(0, Nv, Ni)
    c = np.zeros(Nv, dtype=dtype)
    if random:
        if dtype == 'int64':
            v = np.random.randint(0, 2**32, Ni)
        elif dtype == 'float64':
            v = np.random.random(Ni)
    else:   
        v = np.ones(Ni, dtype=dtype)
    
    timings = []
    for _ in range(trials):
        start = time.time()
        c[i] = v
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    bytes_per_sec = (i.size * i.itemsize * 3) / tavg
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))

def check_correctness(dtype, random):
    Ni = 10**4
    Nv = 10**4
    # make indices unique
    # if indices are non-unique, results of unordered scatter are variable
    npi = np.arange(Ni)
    np.random.shuffle(npi)
    npc = np.zeros(Nv, dtype=dtype)
    aki = ak.array(npi)
    akc = ak.zeros(Nv, dtype=dtype)
    if random:
        if dtype == 'int64':
            npv = np.random.randint(0, 2**32, Ni)
        elif dtype == 'float64':
            npv = np.random.random(Ni)
    else:   
        npv = np.ones(Ni, dtype=dtype)
    akv = ak.array(npv)
    npc[npi] = npv
    akc[aki] = akv
    assert np.allclose(npc, akc.to_ndarray())
    

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Runs and times a scatter operation: C[I] = V in both arkouda and numpy.")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-i', '--index-size', type=int, default=10**8, help='Length of index array (number of scatters to perform)')
    parser.add_argument('-v', '--value-size', type=int, default=10**8, help='Length of array from which values are scattered')
    parser.add_argument('-t', '--trials', type=int, default=6, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='int64', help='Dtype of value array (int64 or float64)')
    parser.add_argument('-r', '--randomize', default=False, action='store_true', help='Use random values instead of ones')
    parser.add_argument('-c', '--check', default=False, action='store_true', help='Run on a small array first to test whether arkouda and numpy results agree')
    args = parser.parse_args()
    if args.dtype not in ('int64', 'float64'):
        raise ValueError("Dtype must be either int64 or float64, not {}".format(args.dtype))
    ak.v = False
    ak.connect(args.hostname, args.port)
    if args.check:
        print("Verifying correctness on small problem... ", end="")
        check_correctness(args.dtype, args.randomize)
        print("CORRECT")
    
    print("size of index array = {:,}".format(args.index_size))
    print("size of values array = {:,}".format(args.value_size))
    print("number of trials = ", args.trials)
    time_ak_scatter(args.index_size, args.value_size, args.trials, args.dtype, args.randomize)
    time_np_scatter(args.index_size, args.value_size, args.trials, args.dtype, args.randomize)
    sys.exit(0)
