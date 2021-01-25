#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak
import random
import string

TYPES = ('int64', 'float64', 'bool', 'str')

def time_ak_bfs_graph(trials:int):
    lines=91
    vertices=17
    col=3
    directed=0
    weighted=1
    Graph=ak.graph_file_read(lines,vertices,col,directed,"kang.gr")
    print("number of vertices ={}".format(Graph.n_vertices))
    print("number of edges ={}".format(Graph.n_edges))
    print("directed graph  ={}".format(Graph.directed))
    print("weighted graph  ={}".format(Graph.weighted))
    print("source of edges   ={}".format(Graph.src))
    print("R dest of edges   ={}".format(Graph.dstR))
    print("dest of edges   ={}".format(Graph.dst))
    print("R source of edges   ={}".format(Graph.srcR))
    print("start   ={}".format(Graph.start_i))
    print("R start   ={}".format(Graph.start_iR))
    print(" neighbour   ={}".format(Graph.neighbour))
    print("R neighbour   ={}".format(Graph.neighbourR))
    print("vertices weight    ={}".format(Graph.v_weight))
    print("edges weight    ={}".format(Graph.e_weight))
    ll,ver = ak.graph_bfs(Graph,4)
    old=-2;
    visit=[]
    for i in range(int(Graph.n_vertices)):
        cur=ll[i]
        if (int(cur)!=int(old)):
            if len(visit) >0:
                print(visit)
            print("current level=",cur,"the vertices at this level are")
            old=cur
            visit=[]
        visit.append(ver[i])
    print(visit)
    
    print("total edges are as follows")
    for i in range(int(Graph.n_edges)):
         print("<",Graph.src[i]," -- ", Graph.dst[i],">")
    '''
    print("total reverse edges are as follows")
    for i in range(int(Graph.n_edges)):
         print("<",Graph.srcR[i]," -- ", Graph.dstR[i],">")
    '''
    timings = []
    for root in range(trials):
        start = time.time()
        level,nodes = ak.graph_bfs(Graph,root)
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials
    print("Average time = {:.4f} sec".format(tavg))
    print("Average Edges = {:.4f} K/s".format(int(Graph.n_edges)/tavg/1024))
    print("Average Vertices = {:.4f} K/s".format(int(Graph.n_vertices)/tavg/1024))
    '''
    #print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))
    '''


def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of suffix array building: C= suffix_array(V)")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-v', '--logvertices', type=int, default=5, help='Problem size: log number of vertices')
    parser.add_argument('-e', '--vedges', type=int, default=2,help='Number of edges per vertex')
    parser.add_argument('-p', '--possibility', type=float, default=0.01,help='Possibility ')
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

    time_ak_bfs_graph(args.trials)
    sys.exit(0)
