#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak
import random
import string

TYPES = ('int64', 'float64', 'bool', 'str')

def time_ak_bfs_graph(lgNv:int, Ne_per_v:int, p:float,directed:int,weighted:int):
    print("Graph BFS")
    cfg = ak.get_config()
    print("server Hostname =",cfg["serverHostname"])
    print("Number of Locales=",cfg["numLocales"])
    print("number of PUs =",cfg["numPUs"])
    print("Max Tasks =",cfg["maxTaskPar"])
    print("Memory =",cfg["physicalMemory"])
    
    lgNv=6
    Ne_per_v=3
    p=0.40
    directed=0
    weighted=0
    '''
    filename="delaunay_n17"
    f = open("../../arkouda/data/delaunay/"+filename+"/"+filename+".mtx")
    Line = f.readline()
    while not(Line[1]>='0' and Line[0]<='9'):
         Line = f.readline()
         b = Line.split(" ")
    edges=int(b[2])
    vertices=max(int(b[0]),int(b[1]))


    '''
    print(lgNv,Ne_per_v,p,directed,weighted)
    start = time.time()
    #Graph=ak.graph_file_read(edges,vertices,2,directed,"../arkouda/data/delaunay/"+filename+"/"+filename+".gr")
    #Graph=ak.graph_file_read(91,20,3,directed,"kang.gr")
    #Graph=ak.graph_file_read(3056,1024,2,directed,"../arkouda/data/"+filename)
    #Graph=ak.graph_file_read(393176,131072,2,directed,"../arkouda/data/"+filename)
    #Graph=ak.graph_file_read(786396,262144,2,directed,"../arkouda/data/delaunay/delaunay_n18.gr")
    Graph=ak.rmat_gen(lgNv, Ne_per_v, p, directed, weighted)
    #Graph=ak.graph_file_read(103689,8276,2,directed,"data/graphs/wiki")
    #Graph=ak.graph_file_read(2981,2888,2,directed,"data/graphs/fb")
    #Graph=ak.graph_file_read(1000,1001,2,directed,"data/line.gr")
    #Graph=ak.graph_file_read(1000,1001,2,directed,"/rhome/zhihui/ArkoudaExtension/arkouda/data/line.gr")
    #Graph=ak.graph_file_read(100,101,2,directed,"/rhome/zhihui/ArkoudaExtension/arkoudabak/data/100.gr")
    #Graph=ak.graph_file_read(10000,10001,2,directed,"data/10000-1.gr")
    #Graph=ak.graph_file_read(100,101,2,directed,"data/100-1.gr")
    #Graph=ak.graph_file_read(2000,1002,2,directed,"data/2.gr")
    #Graph=ak.graph_file_read(3000,1003,2,directed,"data/3.gr")
    #Graph=ak.graph_file_read(150,53,2,directed,"data/3-50.gr")
    end = time.time()
    print("Building RMAT Graph takes {:.4f} seconds".format(end-start))
    print("directed graph  ={}".format(Graph.directed))
    print("number of vertices=", int(Graph.n_vertices))
    print("number of edges=", int(Graph.n_edges))
    print("weighted graph  ={}".format(Graph.weighted))
    print("source of edges   ={}".format(Graph.src))
    print("dest of edges   ={}".format(Graph.dst))
    print("start   ={}".format(Graph.start_i))
    print(" neighbour   ={}".format(Graph.neighbour))
    print("source of edges R  ={}".format(Graph.srcR))
    print("dest of edges R  ={}".format(Graph.dstR))
    print("start R  ={}".format(Graph.start_iR))
    print("neighbour R  ={}".format(Graph.neighbourR))
    print("neighbour size  ={}".format(Graph.neighbour.size))
    print("from src to dst")
    '''
    for i in range(int(Graph.n_edges)):
         print("<",Graph.src[i]," -- ", Graph.dst[i],">")
    print("vertex, neighbour, start")
    for i in range(int(Graph.n_vertices)):
         print("<",i,"--", Graph.neighbour[i],"--", Graph.start_i[i], ">")
    print("from srcR to dstR")
    for i in range(int(Graph.n_edges)):
         print("<",Graph.srcR[i]," -- ", Graph.dstR[i],">")
    print("vertex, neighbourR, startR")
    for i in range(int(Graph.n_vertices)):
         print("<",i,"--", Graph.neighbourR[i],"--", Graph.start_iR[i], ">")
    print("vertices weight    ={}".format(Graph.v_weight))
    print("edges weight    ={}".format(Graph.e_weight))
    '''
    start = time.time()
    deparray = ak.graph_bfs(Graph,0)
    end = time.time()
    print("----------------------")
    print("deparray = ak.graph_bfs(Graph,0)")
    print(deparray)
    print("BFS  the graph takes {:.4f} seconds".format(end-start))
    start = time.time()
    deparray = ak.graph_bfs(Graph,int((int(Graph.n_vertices)-1)/2))
    end = time.time()
    print("----------------------")
    print("deparray = ak.graph_bfs(Graph,",int((int(Graph.n_vertices)-1)/2),")")
    print(deparray)
    print("BFS  the graph takes {:.4f} seconds".format(end-start))
    start = time.time()
    deparray = ak.graph_bfs(Graph,int(Graph.n_vertices)-1)
    end = time.time()
    print("----------------------")
    print("deparray = ak.graph_bfs(Graph,",int(Graph.n_vertices)-1,")")
    print(deparray)
    print("BFS  the graph takes {:.4f} seconds".format(end-start))
    return
    '''
    print("----------------------")
    print("deparray = ak.graph_bfs(Graph,2)")
    print(deparray)
    print("BFS  the graph takes {:.4f} seconds".format(end-start))
    start = time.time()
    deparray = ak.graph_bfs(Graph,int(Graph.n_vertices)-3)
    end = time.time()
    print("deparray = ak.graph_bfs(Graph,Graph.n_vertices-3)")
    print(deparray)
    print("BFS  the graph takes {:.4f} seconds".format(end-start))
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
    print("total reverse edges are as follows")
    for i in range(int(Graph.n_edges)):
         print("<",Graph.srcR[i]," -- ", Graph.dstR[i],">")
    
    '''
    timings = []
    totalV=int(Graph.n_vertices)
    trials=20
    selectroot = np.random.randint(0, totalV-1, trials)
    #selectroot[0]=0
    #for root in range(trials):
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
    #print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))


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
    time_ak_bfs_graph(1,2,3.0,0,0)
    '''
    for i in range(10,25,2):
        for j in range(3,30,4):
           for k in np.arange(0.4,0.7,0.15):
               for d in range(0,2):
                   for w in range(0,1):
                      time_ak_bfs_graph(i,j,k,d,w)
    time_ak_bfs_graph(args.trials)
    sys.exit(0)
    '''
