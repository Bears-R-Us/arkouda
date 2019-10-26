#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak

def time_ak_gather(isize, vsize, trials, dtype, random):
    print(">>> arkouda gather")
    cfg = ak.get_config()
    Ni = isize * cfg["numLocales"]
    Nv = vsize * cfg["numLocales"]
    print("numLocales = {}, num_indices = {:,} ; num_values = {:,}".format(cfg["numLocales"], Ni, Nv))
    # Index vector is always random
    i = ak.randint(0, Nv, Ni)
    if random:
        if dtype == 'int64':
            v = ak.randint(0, 2**32, Nv)
        elif dtype == 'float64':
            v = ak.randint(0, 1, Nv, dtype=ak.float64)
    else:   
        v = ak.ones(Nv, dtype=dtype)
    
    timings = []
    for _ in range(trials):
        start = time.time()
        c = v[i]
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    bytes_per_sec = (c.size * c.itemsize * 3) / tavg
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))

def time_np_gather(Ni, Nv, trials, dtype, random):
    print(">>> numpy gather")
    print("num_indices = {:,} ; num_values = {:,}".format(Ni, Nv))
    # Index vector is always random
    i = np.random.randint(0, Nv, Ni)
    if random:
        if dtype == 'int64':
            v = np.random.randint(0, 2**32, Nv)
        elif dtype == 'float64':
            v = np.random.random(Nv)
    else:   
        v = np.ones(Nv, dtype=dtype)
    
    timings = []
    for _ in range(trials):
        start = time.time()
        c = v[i]
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    bytes_per_sec = (c.size * c.itemsize * 3) / tavg
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))

def check_correctness(dtype, random):
    Ni = 10**4
    Nv = 10**4
    npi = np.random.randint(0, Nv, Ni)
    aki = ak.array(npi)
    if random:
        if dtype == 'int64':
            npv = np.random.randint(0, 2**32, Nv)
        elif dtype == 'float64':
            npv = np.random.random(Nv)
    else:   
        npv = np.ones(Nv, dtype=dtype)
    akv = ak.array(npv)
    npc = npv[npi]
    akc = akv[aki]
    assert np.allclose(npc, akc.to_ndarray())

def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of random gather: C = V[I]")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-i', '--index-size', type=int, default=10**8, help='Length of index array (number of gathers to perform)')
    parser.add_argument('-v', '--value-size', type=int, default=10**8, help='Length of array from which values are gathered')
    parser.add_argument('-t', '--trials', type=int, default=6, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='int64', help='Dtype of value array (int64 or float64)')
    parser.add_argument('-r', '--randomize', default=False, action='store_true', help='Use random values instead of ones')
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
    
    print("size of index array = {:,}".format(args.index_size))
    print("size of values array = {:,}".format(args.value_size))
    print("number of trials = ", args.trials)
    time_ak_gather(args.index_size, args.value_size, args.trials, args.dtype, args.randomize)
    if args.numpy:
        time_np_gather(args.index_size, args.value_size, args.trials, args.dtype, args.randomize)
        print("Verifying agreement between arkouda and NumPy on small problem... ", end="")
        check_correctness(args.dtype, args.randomize)
        print("CORRECT")
        
    sys.exit(0)
