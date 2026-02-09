#!/usr/bin/env python3

import argparse
import time

import numpy as np

import arkouda as ak


OPS = ("intersect", "union", "setdiff", "setxor")
TYPES = (
    "int64",
    "uint64",
)


def time_ak_setops(N_per_locale, trials, dtype, seed):
    print(">>> arkouda segarray {} setops".format(dtype))
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
    if dtype == "int64":
        a = ak.randint(0, 2**32, N, seed=seed)
        b = ak.randint(0, 2**32, N, seed=seed)
        seg_a = ak.SegArray(ak.array([0, len(a)]), ak.concatenate([a, b]))
        c = ak.randint(0, 2**32, N, seed=seed)
        d = ak.randint(0, 2**32, N, seed=seed)
        seg_b = ak.SegArray(ak.array([0, len(c)]), ak.concatenate([c, d]))
    elif dtype == "uint64":
        a = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
        b = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
        seg_a = ak.SegArray(ak.array([0, len(a)]), ak.concatenate[[a, b]])
        c = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
        d = ak.randint(0, 2**32, N, seed=seed, dtype=ak.uint64)
        seg_b = ak.SegArray(ak.array([0, len(c)]), ak.concatenate([c, d]))

    timings = {op: [] for op in OPS}
    results = {}
    for i in range(trials):
        for op in timings.keys():
            fxn = getattr(seg_a, op)
            start = time.time()
            r = fxn(seg_b)
            end = time.time()
            timings[op].append(end - start)
            results[op] = r
    tavg = {op: sum(t) / trials for op, t in timings.items()}

    for op, t in tavg.items():
        print("  {} Average time = {:.4f} sec".format(op, t))
        bytes_per_sec = (a.size * a.itemsize * 2) / t
        print("  {} Average rate = {:.2f} GiB/sec".format(op, bytes_per_sec / 2**30))


NP_OP_MAP = {
    "intersect": "intersect1d",
    "union": "union1d",
    "setdiff": "setdiff1d",
    "setxor": "setxor1d",
}


def check_correctness(dtype, seed):
    N = 10**4
    if seed is not None:
        np.random.seed(seed)
    if dtype == "int64":
        a = np.random.randint(0, 2**32, N)
        b = np.random.randint(0, 2**32, N)
        c = np.random.randint(0, 2**32, N)
        d = np.random.randint(0, 2**32, N)
    if dtype == "uint64":
        a = np.random.randint(0, 2**32, N, dtype=ak.uint64)
        b = np.random.randint(0, 2**32, N, dtype=ak.uint64)
        c = np.random.randint(0, 2**32, N, dtype=ak.uint64)
        d = np.random.randint(0, 2**32, N, dtype=ak.uint64)

    for op in OPS:
        seg_a = ak.SegArray(ak.array([0, len(a)]), ak.array(np.concatenate([a, b])))
        seg_b = ak.SegArray(ak.array([0, len(c)]), ak.array(np.concatenate([c, d])))
        fxn = getattr(np, NP_OP_MAP[op])
        npr0 = fxn(a, c)
        npr1 = fxn(b, d)
        fxn = getattr(seg_a, op)
        seg_r = fxn(seg_b)
        np.isclose(npr0, seg_r[0])
        np.isclose(npr1, seg_r[1])


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run the setops benchmarks: intersect1d, union1d, setdiff1d, setxor1d"
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n", "--size", type=int, default=10**8, help="Problem size: length of arrays A and B"
    )
    parser.add_argument(
        "-t", "--trials", type=int, default=1, help="Number of times to run the benchmark"
    )
    parser.add_argument(
        "-d", "--dtype", default="int64", help="Dtype of array ({})".format(", ".join(TYPES))
    )
    parser.add_argument(
        "--correctness-only",
        default=False,
        action="store_true",
        help="Only check correctness, not performance.",
    )
    parser.add_argument(
        "-s", "--seed", default=None, type=int, help="Value to initialize random number generator"
    )
    return parser


if __name__ == "__main__":
    import sys

    parser = create_parser()
    args = parser.parse_args()
    if args.dtype not in TYPES:
        raise ValueError("Dtype must be {}, not {}".format("/".join(TYPES), args.dtype))

    ak.verbose = False
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        for dtype in TYPES:
            check_correctness(dtype, args.seed)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_setops(args.size, args.trials, args.dtype, args.seed)

    sys.exit(0)
