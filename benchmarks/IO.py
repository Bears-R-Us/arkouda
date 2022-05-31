#!/usr/bin/env python3

import argparse
import os
import time
from glob import glob

import arkouda as ak

TYPES = (
    "int64",
    "float64",
    "uint64",
)


def time_ak_write(N_per_locale, numfiles, trials, dtype, path, seed, parquet, compressed=False):
    if not parquet:
        print(">>> arkouda {} HDF5 write with compressed={}".format(dtype, compressed))
    else:
        print(">>> arkouda {} Parquet write with compressed={}".format(dtype, compressed))
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}, filesPerLoc = {}".format(cfg["numLocales"], N, numfiles))
    if dtype == "int64":
        a = ak.randint(0, 2**32, N, seed=seed)
    elif dtype == "float64":
        a = ak.randint(0, 1, N, dtype=ak.float64, seed=seed)
    elif dtype == "uint64":
        a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)

    writetimes = []
    for i in range(trials):
        for j in range(numfiles):
            start = time.time()
            a.save(f"{path}{j:04}") if not parquet else a.save_parquet(
                f"{path}{j:04}", compressed=compressed
            )
            end = time.time()
            writetimes.append(end - start)
    avgwrite = sum(writetimes) / trials

    print("write Average time = {:.4f} sec".format(avgwrite))

    nb = a.size * a.itemsize * numfiles
    print("write Average rate = {:.2f} GiB/sec".format(nb / 2**30 / avgwrite))


def time_ak_read(N_per_locale, numfiles, trials, dtype, path, seed, parquet):
    if not parquet:
        print(">>> arkouda HDF5 {} read".format(dtype))
    else:
        print(">>> arkouda Parquet {} read".format(dtype))
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}, filesPerLoc = {}".format(cfg["numLocales"], N, numfiles))
    a = ak.array([])

    readtimes = []
    for i in range(trials):
        start = time.time()
        a = ak.read(path + "*")
        end = time.time()
        readtimes.append(end - start)
    avgread = sum(readtimes) / trials

    print("read Average time = {:.4f} sec".format(avgread))

    nb = a.size * a.itemsize
    print("read Average rate = {:.2f} GiB/sec".format(nb / 2**30 / avgread))


def remove_files(path):
    for f in glob(path + "*"):
        os.remove(f)


def check_correctness(dtype, path, seed, parquet, multifile=False):
    N = 10**4
    if dtype == "int64":
        a = ak.randint(0, 2**32, N, seed=seed)
        if multifile:
            b = ak.randint(0, 2**32, N, seed=seed)
    elif dtype == "float64":
        a = ak.randint(0, 1, N, dtype=ak.float64, seed=seed)
        if multifile:
            b = ak.randint(0, 1, N, dtype=ak.float64, seed=seed)
    elif dtype == "uint64":
        a = ak.randint(0, 1, N, dtype=ak.uint64, seed=seed)
        if multifile:
            b = ak.randint(0, 1, N, dtype=ak.uint64, seed=seed)

    a.save(f"{path}{1}") if not parquet else a.save_parquet(f"{path}{1}")
    if multifile:
        b.save(f"{path}{2}") if not parquet else b.save_parquet(f"{path}{2}")

    c = ak.read(path + "*")

    for f in glob(path + "*"):
        os.remove(f)
    if not multifile:
        assert (a == c).all()
    else:
        assert (a == c[0 : a.size]).all()
        assert (b == c[a.size :]).all()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure performance of writing and reading a random array from disk."
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n", "--size", type=int, default=10**8, help="Problem size: length of array to write/read"
    )
    parser.add_argument(
        "-t", "--trials", type=int, default=1, help="Number of times to run the benchmark"
    )
    parser.add_argument(
        "-d", "--dtype", default="int64", help="Dtype of array ({})".format(", ".join(TYPES))
    )
    parser.add_argument(
        "-p",
        "--path",
        default=os.getcwd() + "ak-io-test",
        help="Target path for measuring read/write rates",
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
    parser.add_argument(
        "-q", "--parquet", default=False, action="store_true", help="Perform Parquet operations"
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
        "-l", "--files-per-loc", type=int, default=1, help="Number of files to create per locale"
    )
    parser.add_argument(
        "-c",
        "--compressed",
        default=False,
        action="store_true",
        help="Write with Snappy compression and RLE encoding",
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
            check_correctness(dtype, args.path, args.seed, args.parquet)
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
            args.parquet,
            args.compressed,
        )
    elif args.only_read:
        time_ak_read(
            args.size, args.files_per_loc, args.trials, args.dtype, args.path, args.seed, args.parquet
        )
    else:
        time_ak_write(
            args.size,
            args.files_per_loc,
            args.trials,
            args.dtype,
            args.path,
            args.seed,
            args.parquet,
            args.compressed,
        )
        time_ak_read(
            args.size, args.files_per_loc, args.trials, args.dtype, args.path, args.seed, args.parquet
        )
        remove_files(args.path)

    sys.exit(0)
