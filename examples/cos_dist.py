#!/usr/bin/env python3

import arkouda as ak
import math

# dot product or two arkouda pdarrays
def ak_dot(u, v):
    return ak.sum(u*v)

# magnitude/L_2-norm of arkouda pdarray
def ak_mag2(u):
    return math.sqrt(ak_dot(u,u))

# cosine distance of two arkouda pdarrays
# should function similarly to scipy.spatial.distance.cosine
def ak_cos_dist(u, v):
    return (1.0 - ak_dot(u,v)/(ak_mag2(u)*ak_mag2(v)))

if __name__ == "__main__":
    import argparse, sys, gc, time

    parser = argparse.ArgumentParser(description="Example of cosine distance/similarity in arkouda")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    args = parser.parse_args()

    ak.v = False
    ak.connect(server=args.hostname, port=args.port)

    u1 = [1, 0, 0]
    v1 = [0, 1, 0]
    d1 = ak_cos_dist(ak.array(u1), ak.array(v1))
    print("d1 = ", d1)
    # d1 should be 1.0
    
    u2 = [100, 0, 0]
    d2 = ak_cos_dist(ak.array(u2), ak.array(v1))
    print("d2 = ", d2)
    # d2 should be 1.0
    
    u3 = [1, 1, 0]  
    d3 = ak_cos_dist(ak.array(u3), ak.array(v1))
    print("d3 = ", d3)
    # d3 should be 0.29289321881345254
    
    ak.disconnect()
    
    
