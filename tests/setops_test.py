#!/usr/bin/env python3

import arkouda as ak
import numpy as np
import pandas as pd

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="Runs and times reductions over arrays in both arkouda and numpy.")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**6, help='Problem size: length of array to argsort')

    args = parser.parse_args()
    ak.v = False
    ak.connect(args.hostname, args.port)
    print("size = ",args.size)
    SIZE = args.size
    a = ak.randint(0, 2*SIZE, SIZE)
    b = ak.randint(0, 2*SIZE, SIZE)

    
    set_union = ak.union1d(a,b)
    print("union1d = ", set_union.size,set_union)
    # elements in a or elements in b (or in both a and b)
    passed = ak.all(ak.in1d(set_union,a) | ak.in1d(set_union,b))
    print("union1d passed test: ",passed)
    
    set_intersection = ak.intersect1d(a,b)
    print("intersect1d = ", set_intersection.size,set_intersection)
    # elements in a and elements in b (elements in both a and b)
    passed = ak.all(ak.in1d(set_intersection,a) & ak.in1d(set_intersection,b))
    print("intersect1d passed test: ",passed)
    
    set_difference = ak.setdiff1d(a,b)
    print("setdiff1d = ", set_difference.size,set_difference)
    # elements in a and not in b
    passes = ak.all(ak.in1d(set_difference,a) & ak.in1d(set_difference,b,invert=True))
    print("setdiff1d passed test: ",passed)
    
    set_xor = ak.setxor1d(a,b)
    print("setxor1d = ", set_xor.size,set_xor)
    # elements NOT in the intersection of a and b
    passes = ak.all(ak.in1d(set_xor, set_intersection, invert=True))
    print("setxor1d passed test: ",passed)

    ak.disconnect()
    
    
