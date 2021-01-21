#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak
import random
import string

TYPES = ('int64', 'float64', 'bool', 'str')

def time_ak_rmat_graph(lgNv, Ne_per_v, p, perm):
    print(">>> arkouda rmat graph")
    cfg = ak.get_config()
    Nv =  cfg["numLocales"]
    print("numLocales = {}".format(cfg["numLocales"]))
    Graph = ak.rmat_gen(lgNv, Ne_per_v, p, perm)
    print("number of vertices ={}".format(Graph.n_vertices))
    print("number of edges ={}".format(Graph.n_edges))
    print("directed graph  ={}".format(Graph.directed))
    print("source of edges   ={}".format(Graph.src))
    print("dest of edges   ={}".format(Graph.dst))
    print("vertices weight    ={}".format(Graph.v_weight))
    print("edges weight    ={}".format(Graph.e_weight))
    print("neighbour   ={}".format(Graph.neighbour))
    return
    timings = []
    for _ in range(trials):
        start = time.time()
        c=ak.suffix_array(v)
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    if dtype == 'str':
        offsets_transferred = 0 * c.offsets.size * c.offsets.itemsize
        bytes_transferred = (c.bytes.size * c.offsets.itemsize) + (0 * c.bytes.size)
        bytes_per_sec = (offsets_transferred + bytes_transferred) / tavg
    else:
        print("Wrong data type")
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))


def suffixArray(s):
    suffixes = [(s[i:], i) for i in range(len(s))]
    suffixes.sort(key=lambda x: x[0])
    sa= [s[1] for s in suffixes]
    #sa.insert(0,len(sa))
    return sa

def time_np_sa(vsize, strlen, trials, dtype):
    s=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(strlen))
    timings = []
    for _ in range(trials):
        start = time.time()
        sa=suffixArray(s)
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials
    print("Average time = {:.4f} sec".format(tavg))
    if dtype == 'str':
        offsets_transferred = 0
        bytes_transferred = len(s)
        bytes_per_sec = (offsets_transferred + bytes_transferred) / tavg
    else:
        print("Wrong data type")
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))

def check_correctness( vsize,strlen, trials, dtype):
    Ni = strlen
    Nv = vsize

    v = ak.random_strings_uniform(1, Ni, Nv)
    c=ak.suffix_array(v)
    for k in range(Nv):
        s=v[k]
        sa=suffixArray(s)
        aksa=c[k]
#        _,tmp=c[k].split(maxsplit=1)
#        aksa=tmp.split()
#        intaksa  = [int(numeric_string) for numeric_string in aksa]
#        intaksa  = aksa[1:-1]
#        print(sa)
#        print(intaksa)
        assert (sa==aksa)


def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of suffix array building: C= suffix_array(V)")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-v', '--logvertices', type=int, default=7, help='Problem size: log number of vertices')
    parser.add_argument('-e', '--vedges', type=int, default=2,help='Number of edges per vertex')
    parser.add_argument('-p', '--possibility', type=float, default=0.03,help='Possibility ')
    parser.add_argument('-t', '--trials', type=int, default=6, help='Number of times to run the benchmark')
    parser.add_argument('-m', '--perm', type=int, default=0 , help='if permutation ')
    parser.add_argument('--numpy', default=False, action='store_true', help='Run the same operation in NumPy to compare performance.')
    parser.add_argument('--correctness-only', default=False, action='store_true', help='Only check correctness, not performance.')
    return parser


    
if __name__ == "__main__":
    import sys
    parser = create_parser()
    args = parser.parse_args()
    ak.verbose = False
    ak.connect(args.hostname, args.port)

    '''
    if args.correctness_only:
        check_correctness(args.number, args.size, args.trials, args.dtype)
        print("CORRECT")
        sys.exit(0)
    '''

    time_ak_rmat_graph(args.logvertices, args.vedges, args.possibility, args.perm)
    sys.exit(0)
