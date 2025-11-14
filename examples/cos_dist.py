#!/usr/bin/env python3

import math

import arkouda as ak


# dot product or two arkouda pdarrays
def ak_dot(u, v):
    if isinstance(u, ak.pdarray) and isinstance(v, ak.pdarray):
        return ak.sum(u * v)
    else:
        raise TypeError("u and v must be pdarrays")


# magnitude/L_2-norm of arkouda pdarray
def ak_mag2(u):
    if isinstance(u, ak.pdarray):
        return math.sqrt(ak_dot(u, u))
    else:
        raise TypeError("u must be a pdarray")


# cosine distance of two arkouda pdarrays
# should function similarly to scipy.spatial.distance.cosine
def ak_cos_dist(u, v):
    if isinstance(u, ak.pdarray) and isinstance(v, ak.pdarray):
        return 1.0 - ak_dot(u, v) / (ak_mag2(u) * ak_mag2(v))
    else:
        raise TypeError("u and v must be pdarrays")


if __name__ == "__main__":
    import argparse
    import gc
    import sys
    import time

    import numpy as np

    from scipy.spatial import distance

    parser = argparse.ArgumentParser(description="Example of cosine distance/similarity in arkouda")
    parser.add_argument("--server", default="localhost", help="server/Hostname of arkouda server")
    parser.add_argument("--port", type=int, default=5555, help="Port of arkouda server")
    args = parser.parse_args()

    ak.v = False
    ak.connect(server=args.server, port=args.port)

    u1 = [1, 0, 0]
    v1 = [0, 1, 0]
    d1 = ak_cos_dist(ak.array(u1), ak.array(v1))
    print("d1 = ", d1)
    # d1 should be 1.0
    assert np.allclose(d1, distance.cosine(u1, v1))

    u2 = [100, 0, 0]
    d2 = ak_cos_dist(ak.array(u2), ak.array(v1))
    print("d2 = ", d2)
    # d2 should be 1.0
    assert np.allclose(d2, distance.cosine(u2, v1))

    u3 = [1, 1, 0]
    d3 = ak_cos_dist(ak.array(u3), ak.array(v1))
    print("d3 = ", d3)
    # d3 should be 0.29289321881345254
    assert np.allclose(d3, distance.cosine(u3, v1))

    ak.disconnect()
