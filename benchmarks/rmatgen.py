#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak
import random
import string

TYPES = ('int64', 'float64', 'bool', 'str')

def time_ak_rmat_graph(lgNv, Ne_per_v, p, directed, weighted):
    print(">>> arkouda rmat graph")
    cfg = ak.get_config()
    Nv =  cfg["numLocales"]
    print("numLocales = {}".format(cfg["numLocales"]))
    Graph = ak.rmat_gen(lgNv, Ne_per_v, p, directed, weighted)
    print("number of vertices ={}".format(Graph.n_vertices))
    print("number of edges ={}".format(Graph.n_edges))
    print("directed graph  ={}".format(Graph.directed))
    print("source of edges   ={}".format(Graph.src))
    print("dest of edges   ={}".format(Graph.dst))
    print("start   ={}".format(Graph.start_i))
    print("neighbour   ={}".format(Graph.neighbour))
    #print("vertices weight    ={}".format(Graph.v_weight))
    #print("edges weight    ={}".format(Graph.e_weight))
    timings = []
    for _ in range(trials):
        start = time.time()
        Graph = ak.rmat_gen(lgNv, Ne_per_v, p, perm)
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))



def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of suffix array building: C= suffix_array(V)")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-v', '--logvertices', type=int, default=5, help='Problem size: log number of vertices')
    parser.add_argument('-e', '--vedges', type=int, default=2,help='Number of edges per vertex')
    parser.add_argument('-p', '--possibility', type=float, default=0.01,help='Possibility ')
    parser.add_argument('-t', '--trials', type=int, default=6, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--directed', type=int, default=0 , help='if directed ')
    parser.add_argument('-w', '--weighted', type=int, default=0 , help='if weighted ')
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

    time_ak_rmat_graph(args.logvertices, args.vedges, args.possibility, args.directed,args.weighted)
    sys.exit(0)
