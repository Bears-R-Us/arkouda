#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak
import random
import string

TYPES = ('int64', 'float64', 'bool', 'str')

def time_ak_bfs_graph(lgNv:int, Ne_per_v:int, p:float,directed:int,weighted:int,trials):
    print("Graph BFS")
    cfg = ak.get_config()
    print("server Hostname =",cfg["serverHostname"])
    print("Number of Locales=",cfg["numLocales"])
    print("number of PUs =",cfg["numPUs"])
    print("Max Tasks =",cfg["maxTaskPar"])
    print("Memory =",cfg["physicalMemory"])
    
    print(lgNv,Ne_per_v,p,directed,weighted)
    start = time.time()
    #Graph=ak.graph_file_read(91,20,3,directed,"kang.gr")
    Graph=ak.rmat_gen(lgNv, Ne_per_v, p, directed, weighted)
    end = time.time()
    print("Building RMAT Graph takes {:.4f} seconds".format(end-start))
    print("directed graph  ={}".format(Graph.directed))
    print("number of vertices=", int(Graph.n_vertices))
    print("number of edges=", int(Graph.n_edges))
    print("weighted graph  ={}".format(Graph.weighted))
    timings = []
    totalV=int(Graph.n_vertices)
    trials=20
    selectroot = np.random.randint(0, totalV-1, trials)
    for root in selectroot:
        start = time.time()
        _ = ak.graph_bfs(Graph,int(root))
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials
    print("Average BFS time = {:.4f} s for {} executions".format(tavg,trials))
    print("number of vertices ={}".format(Graph.n_vertices))
    print("number of edges ={}".format(Graph.n_edges))
    print("Average BFS Edges = {:.4f} M/s".format(int(Graph.n_edges)/tavg/1024/1024))
    print("Average BFS Vertices = {:.4f} M/s".format(int(Graph.n_vertices)/tavg/1024/1024))
    print("Ne_per_v=",Ne_per_v, " p=" ,p)


def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of suffix array building: C= suffix_array(V)")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-v', '--logvertices', type=int, default=5, help='Problem size: log number of vertices')
    parser.add_argument('-e', '--vedges', type=int, default=2,help='Number of edges per vertex')
    parser.add_argument('-p', '--possibility', type=float, default=0.01,help='Possibility ')
    parser.add_argument('-d', '--directed', type=int, default=0,help='Directed Graph ')
    parser.add_argument('-w', '--weighted', type=int, default=0,help='Weighted Graph ')
    parser.add_argument('-t', '--trials', type=int, default=6, help='Number of times to run the benchmark')
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
    time_ak_bfs_graph(args.logvertices,args.vedges,args.possibility,args.directed,args.weighted,args.trials)
