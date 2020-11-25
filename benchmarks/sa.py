#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak

TYPES = ('int64', 'float64', 'bool', 'str')

def time_ak_sa( vsize, trials, dtype):
    print(">>> arkouda suffix array")
    cfg = ak.get_config()
    Nv = vsize * cfg["numLocales"]
    print("numLocales = {},  num of strings  = {:,}".format(cfg["numLocales"], Nv))
#    v = ak.random_strings_uniform(90000000, 100000000, Nv)
    v = ak.random_strings_uniform(1, 16, Nv)
    c=ak.suffix_array(v)
    print("size of suffix array={}".format(c.bytes.size))    
#    print("All the random strings are as follows")
    for k in range(vsize):
       print("the {} th random tring ={}".format(k,v[k]))    
       print("the {} th suffix array ={}".format(k,c[k]))    
       print("")
#    print(v)    
    timings = []
    for _ in range(trials):
        start = time.time()
        ak.suffix_array(v)
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    if dtype == 'str':
        offsets_transferred = 3 * c.offsets.size * c.offsets.itemsize
        bytes_transferred = (c.offsets.size * c.offsets.itemsize) + (2 * c.bytes.size)
        bytes_per_sec = (offsets_transferred + bytes_transferred) / tavg
    else:
        bytes_per_sec = (c.size * c.itemsize * 3) / tavg
#    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))

def time_np_sa(Ni, Nv, trials, dtype, random):
    print("to be done")

def check_correctness(dtype, random):
    print("to be done")

def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of suffix array building: C= V")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-v', '--value-size', type=int, help='Length of array from which values are gathered')
    parser.add_argument('-t', '--trials', type=int, default=6, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='str', help='Dtype of value array ({})'.format(', '.join(TYPES)))
    return parser
    
if __name__ == "__main__":
    import sys
    parser = create_parser()
    args = parser.parse_args()
    args.value_size = args.size if args.value_size is None else args.value_size
    if args.dtype not in TYPES:
        raise ValueError("Dtype must be {}, not {}".format('/'.join(TYPES), args.dtype))
    ak.verbose = False
    ak.connect(args.hostname, args.port)

    print("size of values array = {:,}".format(args.value_size))
    print("number of trials = ", args.trials)
    time_ak_sa( args.value_size, args.trials, args.dtype)
        
    sys.exit(0)
