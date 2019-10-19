#!/usr/bin/env python3

# this is all kinda of clunky because...
# we are using an edgelist/tuple formulation...
# but at least you can do it... ;-)
#
# generate and rmat graph
# make the graph undirected/symmetric
# find the graph's connected components using bfs

import arkouda as ak

def gen_rmat_edges(lgNv, Ne_per_v, p, perm=False):
    # number of vertices
    Nv = 2**lgNv
    # number of edges
    Ne = Ne_per_v * Nv
    # probabilities
    a = p
    b = (1.0 - a)/ 3.0
    c = b
    d = b
    # init edge arrays
    ii = ak.ones(Ne,dtype=ak.int64)
    jj = ak.ones(Ne,dtype=ak.int64)
    # quantites to use in edge generation loop
    ab = a+b
    c_norm = c / (c + d)
    a_norm = a / (a + b)
    # generate edges
    for ib in range(1,lgNv):
        ii_bit = (ak.randint(0,1,Ne,dtype=ak.float64) > ab)
        jj_bit = (ak.randint(0,1,Ne,dtype=ak.float64) > (c_norm * ii_bit + a_norm * (~ ii_bit)))
        ii = ii + ((2**(ib-1)) * ii_bit)
        jj = jj + ((2**(ib-1)) * jj_bit)
    # sort all based on ii and jj using coargsort
    # all edges should be sorted based on both vertices of the edge
    iv = ak.coargsort((ii,jj))
    # permute into sorted order
    ii = ii[iv] # permute first vertex into sorted order
    jj = jj[iv] # permute second vertex into sorted order
    # to premute/rename vertices
    if perm:
        # generate permutation for new vertex numbers(names)
        ir = ak.argsort(ak.randint(0,1,Nv,dtype=ak.float64))
        # renumber(rename) vertices
        ii = ir[ii] # rename first vertex
        jj = ir[jj] # rename second vertex
    #
    # maybe: remove edges which are self-loops???
    #    
    # return pair of pdarrays
    return (ii,jj)

# src and dst pdarrays hold the edge list
# seeds pdarray with starting vertices/seeds
def bfs(src,dst,seeds,printLayers=False):
    # holds vertices in the current layer of the bfs
    Z = ak.unique(seeds)
    # holds the visited vertices
    V = ak.unique(Z) # holds vertices in Z to start with
    # frontiers
    F = [Z]
    while Z.size != 0:
        if printLayers:
            print("Z.size = ",Z.size," Z = ",Z)
        fZv = ak.in1d(src,Z) # find src vertex edges 
        W = ak.unique(dst[fZv]) # compress out dst vertices to match and make them unique
        Z = ak.setdiff1d(W,V) # subtract out vertices already visited
        V = ak.union1d(V,Z) # union current frontier into vertices already visited
        F.append(Z)
    return (F,V)

# src pdarray holding source vertices
# dst pdarray holding destination vertices
# printCComp flag to print the connected components as they are found
#
# edges needs to be symmetric/undirected
def conn_comp(src, dst, printCComp=False, printLayers=False):
    unvisited = ak.unique(src)
    if printCComp: print("unvisited size = ", unvisited.size, unvisited)
    components = []
    while unvisited.size > 0:
        # use lowest numbered vertex as representative vertex 
        rep_vertex = unvisited[0]
        # bfs from rep_vertex
        layers,visited = bfs(src,dst,ak.array([rep_vertex]),printLayers)
        # add verticies in component to list of components
        components.append(visited)
        # subtract out visited from unvisited vertices
        unvisited = ak.setdiff1d(unvisited,visited)
        if printCComp: print("  visited size = ", visited.size, visited)
        if printCComp: print("unvisited size = ", unvisited.size, unvisited)
    return components

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse, sys, gc, math, time
    
    parser = argparse.ArgumentParser(description="Generates an rmat structured spare matrix as tuples(ii,jj)")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('--lgNv', type=int, default=20, help='problem scale: log_2(Vertices)')
    parser.add_argument('--Ne_per_v', type=int, default=2, help='number of edges per vertex')
    parser.add_argument('--prob', type=float, default=0.01, help='prob of quadrant-0')
    parser.add_argument('--perm', default=False, action='store_true', help='permute vertex indices/names')
    parser.add_argument('--pl', default=False, action='store_true', help='print layers in bfs')
    parser.add_argument('--pc', default=False, action='store_true', help='print connected comp as they are found')
    
    args = parser.parse_args()

    ak.v = False
    ak.connect(args.hostname, args.port)

    print((args.lgNv, args.Ne_per_v, args.prob, args.perm, args.pl))
    (ii,jj) = gen_rmat_edges(args.lgNv, args.Ne_per_v, args.prob, perm=args.perm)
    
    print("ii = ", (ii.size, ii))
    print("ii(min,max) = ", (ii.min(), ii.max()))
    print("jj = ", (jj.size, jj))
    print("jj(min,max) = ", (jj.min(), jj.max()))

    # make graph undirected/symmetric
    # graph needs to undirected for connected components to work
    src = ak.concatenate((ii,jj))
    dst = ak.concatenate((jj,ii))

    print("src = ", (src.size, src))
    print("src(min,max) = ", (src.min(), src.max()))
    print("dst = ", (dst.size, dst))
    print("dst(min,max) = ", (dst.min(), dst.max()))

    # find components using BFS
    components = conn_comp(src, dst, printCComp=args.pc, printLayers=args.pl)
    print("number of components = ",len(components))
    print("representative vertices = ",[c[0] for c in components])
    
    ak.disconnect()
