#!/usr/bin/env python3

import argparse

from gather import check_correctness, time_ak_gather, time_np_gather

import arkouda as ak

TYPES = ("str",)


def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of random gather: C = V[I]")
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n",
        "--size",
        type=int,
        default=10**8,
        help="Problem size: length of index and gather arrays",
    )
    parser.add_argument(
        "-i",
        "--index-size",
        type=int,
        help="Length of index array (number of gathers to perform)",
    )
    parser.add_argument(
        "-v",
        "--value-size",
        type=int,
        help="Length of array from which values are gathered",
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=1,
        help="Number of times to run the benchmark",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        default="str",
        help="Dtype of value array ({})".format(", ".join(TYPES)),
    )
    parser.add_argument(
        "-r",
        "--randomize",
        default=True,
        action="store_true",
        help="Use random values instead of ones",
    )
    parser.add_argument(
        "--numpy",
        default=False,
        action="store_true",
        help="Run the same operation in NumPy to compare performance.",
    )
    parser.add_argument(
        "--correctness-only",
        default=False,
        action="store_true",
        help="Only check correctness, not performance.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=None,
        type=int,
        help="Value to initialize random number generator",
    )
    return parser


if __name__ == "__main__":
    import sys

    parser = create_parser()
    args = parser.parse_args()
    args.index_size = args.size if args.index_size is None else args.index_size
    args.value_size = args.size if args.value_size is None else args.value_size
    if args.dtype not in TYPES:
        raise ValueError("Dtype must be {}, not {}".format("/".join(TYPES), args.dtype))
    ak.verbose = False
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        for dtype in TYPES:
            check_correctness(dtype, args.randomize, args.seed)
        sys.exit(0)

    print("size of index array = {:,}".format(args.index_size))
    print("size of values array = {:,}".format(args.value_size))
    print("number of trials = ", args.trials)
    time_ak_gather(
        args.index_size,
        args.value_size,
        args.trials,
        args.dtype,
        args.randomize,
        args.seed,
    )
    if args.numpy:
        time_np_gather(
            args.index_size,
            args.value_size,
            args.trials,
            args.dtype,
            args.randomize,
            args.seed,
        )
        print("Verifying agreement between arkouda and NumPy on small problem... ", end="")
        check_correctness(args.dtype, args.randomize, args.seed)
        print("CORRECT")

    sys.exit(0)
