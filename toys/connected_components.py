#!/usr/bin/env python3
import arkouda as ak

# generate rmat graph edge-list as two pdarrays
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

ak.connect(server="localhost", port=5555)
(ii,jj) = gen_rmat_edges(20, 2, 0.03, perm=True)
src = ak.concatenate((ii,jj))# make graph undirected/symmetric
dst = ak.concatenate((jj,ii))# graph needs to undirected for connected components to work
components = conn_comp(src, dst, printCComp=False, printLayers=False) # find components
print("number of components = ",len(components))
print("representative vertices = ",[c[0] for c in components])
ak.shutdown()
