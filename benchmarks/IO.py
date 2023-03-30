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
    "str"
)
COMPRESSIONS = (
    "none",
    "snappy",
    "gzip",
    "brotli",
    "zstd",
    "lz4"
)


def time_ak_write(N_per_locale, numfiles, trials, dtype, path, seed, parquet, comps=None):
    if comps is None or comps == [""]:
        comps = COMPRESSIONS

    if not parquet:
        print(">>> arkouda {} HDF5 write with compression={}".format(dtype, comps))
    else:
        print(">>> arkouda {} Parquet write with compression={}".format(dtype, comps))
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}, filesPerLoc = {}".format(cfg["numLocales"], N, numfiles))
    if dtype == "int64":
        a = ak.randint(0, 2**32, N, seed=seed)
    elif dtype == "float64":
        a = ak.randint(0, 1, N, dtype=ak.float64, seed=seed)
    elif dtype == "uint64":
        a = ak.randint(0, 2**32, N, dtype=ak.uint64, seed=seed)
    elif dtype == "str":
        a = ak.random_strings_uniform(1, 16, N, seed=seed)

    times = {}
    if parquet:
        for comp in comps:
            if comp in COMPRESSIONS:
                writetimes = []
                for i in range(trials):
                    for j in range(numfiles):
                        start = time.time()
                        a.to_parquet(
                            f"{path}{comp}{j:04}", compression=None if comp == "none" else comp
                        )
                        end = time.time()
                        writetimes.append(end - start)
                times[comp] = sum(writetimes) / trials
    else:
        writetimes = []
        for i in range(trials):
            for j in range(numfiles):
                start = time.time()
                a.to_hdf(f"{path}{j:04}")
                end = time.time()
                writetimes.append(end - start)
        times["HDF5"] = sum(writetimes) / trials

    nb = a.size * a.itemsize * numfiles
    for key in times.keys():
        print("write Average time {} = {:.4f} sec".format(key, times[key]))
        print("write Average rate {} = {:.2f} GiB/sec".format(key, nb / 2**30 / times[key]))


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
        a = ak.read_hdf(path + "*") if not parquet else ak.read_parquet(path + "*")
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
    elif dtype == "str":
        a = ak.random_strings_uniform(1, 16, N, seed=seed)
        if multifile:
            b = ak.random_strings_uniform(1, 16, N, seed=seed)

    a.to_hdf(f"{path}{1}") if not parquet else a.to_parquet(f"{path}{1}")
    if multifile:
        b.to_hdf(f"{path}{2}") if not parquet else b.to_parquet(f"{path}{2}")

    c = ak.read_hdf(path + "*") if not parquet else ak.read_parquet(path + "*")

    remove_files(path)
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
        default=os.path.join(os.getcwd(), "ak-io-test"),
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
        "--compression",
        default="",
        action="store",
        help="Compression types to run Parquet benchmarks against. Comma delimited list (NO SPACES) allowing "
             "for multiple. Accepted values: none, snappy, gzip, brotli, zstd, and lz4"
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
            comp_types,
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
            comp_types,
        )
        time_ak_read(
            args.size, args.files_per_loc, args.trials, args.dtype, args.path, args.seed, args.parquet
        )
        remove_files(args.path)

    sys.exit(0)
