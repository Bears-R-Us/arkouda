#!/usr/bin/env python3

import argparse
import os

from IO import (
    COMPRESSIONS,
    FileFormat,
    check_correctness,
    remove_files,
    time_ak_read,
    time_ak_write,
)

import arkouda as ak

from server_util.test.server_test_util import get_default_temp_directory


TYPES = ("int64", "float64", "uint64", "str")


def create_parser():
    parser = argparse.ArgumentParser(description="Measure performance of Parquet reads/writes.")
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n",
        "--size",
        type=int,
        default=10**6,
        help="Problem size: length of array to read/write",
    )
    parser.add_argument(
        "--fixed-size",
        type=int,
        default=-1,
        help="Fixed size length of string for Parquet",
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
        default="int64",
        help="Dtype of array ({})".format(", ".join(TYPES)),
    )
    parser.add_argument(
        "--correctness-only",
        default=False,
        action="store_true",
        help="Only check correctness, not performance.",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=os.path.join(get_default_temp_directory(), "ak-io-test"),
        help="Target path for measuring read/write rates",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=None,
        type=int,
        help="Value to initialize random number generator",
    )
    parser.add_argument(
        "-w",
        "--only-write",
        default=False,
        action="store_true",
        help="Only write the files; files will not be removed",
    )
    parser.add_argument(
        "-r",
        "--only-read",
        default=False,
        action="store_true",
        help="Only read the files; files will not be removed",
    )
    parser.add_argument(
        "-f",
        "--only-delete",
        default=False,
        action="store_true",
        help="Only delete files created from writing with this benchmark",
    )
    parser.add_argument(
        "-l",
        "--files-per-loc",
        type=int,
        default=1,
        help="Number of files to create per locale",
    )
    parser.add_argument(
        "-c",
        "--compression",
        default="",
        action="store",
        help="Compression types to run Parquet benchmarks against. "
        "Comma delimited list (NO SPACES) allowing "
        "for multiple. Accepted values: none, snappy, gzip, brotli, zstd, and lz4",
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
    comp_str = args.compression
    comp_types = COMPRESSIONS if comp_str == "" else comp_str.lower().split(",")

    if args.correctness_only:
        for dtype in TYPES:
            check_correctness(dtype, args.path, args.seed, FileFormat.PARQUET)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)

    if args.only_write:
        time_ak_write(
            args.size,
            args.files_per_loc,
            args.trials,
            args.dtype,
            args.path,
            args.seed,
            FileFormat.PARQUET,
            comp_types,
            args.fixed_size,
        )
    elif args.only_read:
        time_ak_read(
            args.size,
            args.files_per_loc,
            args.trials,
            args.dtype,
            args.path,
            FileFormat.PARQUET,
            comp_types,
            args.fixed_size,
        )
    else:
        time_ak_write(
            args.size,
            args.files_per_loc,
            args.trials,
            args.dtype,
            args.path,
            args.seed,
            FileFormat.PARQUET,
            comp_types,
            args.fixed_size,
        )
        time_ak_read(
            args.size,
            args.files_per_loc,
            args.trials,
            args.dtype,
            args.path,
            FileFormat.PARQUET,
            comp_types,
            args.fixed_size,
        )
        remove_files(args.path)

    sys.exit(0)
