#!/usr/bin/env python3

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse, sys, gc, math, time
    
    parser = argparse.ArgumentParser(description="Generates an rmat structured spare matrix as tuples(ii,jj)")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('--lgNv', type=int, default=10, help='problem scale: log_2(Vertices)')
    parser.add_argument('--Ne_per_v', type=int, default=10, help='number of edges per vertex')
    parser.add_argument('--prob', type=float, default=0.01, help='prob of quadrant-0')
    parser.add_argument('--perm', default=False, action='store_true', help='permute vertex indices/names')
    args = parser.parse_args()

    ak.v = False
    ak.connect(args.hostname, args.port)

    print((args.lgNv, args.Ne_per_v, args.prob, args.perm))
    (ii,jj) = gen_rmat_edges(args.lgNv, args.Ne_per_v, args.prob, perm=args.perm)
    
    print("ii = ", (ii.size, ii))
    print("ii(min,max) = ", (ii.min(), ii.max()))
    print("jj = ", (jj.size, jj))
    print("jj(min,max) = ", (jj.min(), jj.max()))

    nda_ii = ii.to_ndarray() # convert to ndarray for plotting
    nda_jj = jj.to_ndarray() # convert to ndarray for plotting
    plt.scatter(nda_ii,nda_jj)
    plt.show()

    df = {"ii":ii, "jj":jj}
    
    grps = ak.GroupBy(ii)
    ukeys,cts = grps.count()
    print("counts",(cts.min(),cts.max()))
    nBins = ak.max(cts)
    nda_cts = cts.to_ndarray() # convert to ndarray for plotting
    plt.hist(nda_cts,bins=nBins)
    plt.yscale('log')
    plt.show()
    
    ukeys,nu = grps.nunique(jj)
    print("nunique",(nu.min(), nu.max()))
    nBins = nu.max()
    nda_nu = nu.to_ndarray() # convert to ndarray for plotting
    plt.hist(nda_nu,bins=nBins)
    plt.yscale('log')
    plt.show()

    ak.disconnect()
