#!/usr/bin/env python3

import argparse
from enum import Enum
import os
import time
from glob import glob

import arkouda as ak
import numpy as np

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

class FileFormat(Enum):
    HDF5 = 1
    PARQUET = 2
    CSV = 3

def time_ak_write(N_per_locale, numfiles, trials, dtype, path, seed, fileFormat, comps=None, fixed_size=-1):
    if comps is None or comps == [""]:
        comps = COMPRESSIONS

    file_format_actions = {
        FileFormat.HDF5: ">>> arkouda {} HDF5 write with compression={}".format(dtype, comps),
        FileFormat.PARQUET: ">>> arkouda {} Parquet write with compression={}".format(dtype, comps),
        FileFormat.CSV: ">>> arkouda {} CSV write".format(dtype)
    }
    print(file_format_actions.get(fileFormat, "Invalid file format"))

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
        if fixed_size > 0:
            a = ak.random_strings_uniform(fixed_size, fixed_size+1, N, seed=seed)
        else:
            a = ak.random_strings_uniform(1, 16, N, seed=seed)

    times = {}
    if fileFormat == FileFormat.PARQUET:
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
    elif fileFormat == FileFormat.HDF5:
        writetimes = []
        for i in range(trials):
            for j in range(numfiles):
                start = time.time()
                a.to_hdf(f"{path}{j:04}")
                end = time.time()
                writetimes.append(end - start)
        times["HDF5"] = sum(writetimes) / trials
    elif fileFormat == FileFormat.CSV:
        writetimes = []
        for i in range(trials):
            for j in range(numfiles):
                start = time.time()
                a.to_csv(f"{path}{j:04}")
                end = time.time()
                writetimes.append(end - start)
        times["CSV"] = sum(writetimes) / trials
    else:
        raise ValueError("Invalid file format")

    nb = a.size * a.itemsize * numfiles if dtype != 'str' else a.nbytes * numfiles
    for key in times.keys():
        print("write Average time {} = {:.4f} sec".format(key, times[key]))
        print("write Average rate {} = {:.4f} GiB/sec".format(key, nb / 2**30 / times[key]))


def time_ak_read(N_per_locale, numfiles, trials, dtype, path, fileFormat, comps=None, fixed_size=-1):
    if comps is None or comps == [""]:
        comps = COMPRESSIONS

    file_format_actions = {
        FileFormat.HDF5: ">>> arkouda {} HDF5 read with compression={}".format(dtype, comps),
        FileFormat.PARQUET: ">>> arkouda {} Parquet read with compression={}".format(dtype, comps),
        FileFormat.CSV: ">>> arkouda {} CSV read".format(dtype)
    }
    print(file_format_actions.get(fileFormat, "Invalid file format"))

    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}, filesPerLoc = {}".format(cfg["numLocales"], N, numfiles))
    a = ak.array([])

    times = {}
    if fileFormat == FileFormat.PARQUET:
        if fixed_size < 1:
            for comp in COMPRESSIONS:
                if comp in comps:
                    readtimes = []
                    for i in range(trials):
                        start = time.time()
                        a = ak.read_parquet(path + comp + "*").popitem()[1]
                        end = time.time()
                        readtimes.append(end - start)
                    times[comp] = sum(readtimes) / trials
        else:
            for comp in COMPRESSIONS:
                if comp in comps:
                    readtimes = []
                    for i in range(trials):
                        start = time.time()
                        a = ak.read_parquet(path + comp + "*",fixed_len=fixed_size).popitem()[1]
                        end = time.time()
                        readtimes.append(end - start)
                    times[comp] = sum(readtimes) / trials

    elif fileFormat == FileFormat.HDF5:
        readtimes = []
        for i in range(trials):
            start = time.time()
            a = ak.read_hdf(path + "*").popitem()[1]
            end = time.time()
            readtimes.append(end - start)
        times["HDF5"] = sum(readtimes) / trials
    elif fileFormat == FileFormat.CSV:
        readtimes = []
        for i in range(trials):
            start = time.time()
            a = ak.read_csv(path + "*").popitem()[1]
            end = time.time()
            readtimes.append(end - start)
        times["CSV"] = sum(readtimes) / trials
    else:
        raise ValueError("Invalid file format")

    nb = a.size * a.itemsize if dtype != 'str' else a.nbytes
    for key in times.keys():
        print("read Average time {} = {:.4f} sec".format(key, times[key]))
        print("read Average rate {} = {:.4f} GiB/sec".format(key, nb / 2**30 / times[key]))


def remove_files(path):
    for f in glob(path + "*"):
        os.remove(f)


def check_correctness(dtype, path, seed, fileFormat, multifile=False):
    N = 10**4
    b = None
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
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    file_format_actions = {
        FileFormat.HDF5: (a.to_hdf, b.to_hdf if b is not None else None, ak.read_hdf),
        FileFormat.PARQUET: (a.to_parquet, b.to_parquet if b is not None else None, ak.read_parquet),
        FileFormat.CSV: (a.to_csv, b.to_csv if b is not None else None, ak.read_csv)
    }

    if fileFormat in file_format_actions:
        write_a, write_b, read_c = file_format_actions.get(fileFormat)
    else:
        raise ValueError(f"Invalid file format: {fileFormat}")


    write_a(f"{path}{1}")
    if multifile:
        write_b(f"{path}{2}")

    c = read_c(path + "*").popitem()[1]
    remove_files(path)

    if dtype == "float64":
        assert np.allclose(a.to_ndarray(), c[0 : a.size].to_ndarray()) # Slice is full array when single file
        if multifile:
            assert np.allclose(b.to_ndarray(), c[a.size :].to_ndarray())
    else:
        assert (a == c[0 : a.size]).all() # Slice is full array when single file
        if multifile:
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
        "--fixed-size", type=int, default=-1, help="Fixed size length of string for Parquet"
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-q", "--parquet", default=False, action="store_true", help="Perform Parquet operations"
    )
    group.add_argument(
        "-v", "--csv", default=False, action="store_true", help="Perform CSV operations"
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

    fileFormat = FileFormat.CSV if args.csv else FileFormat.PARQUET if args.parquet else FileFormat.HDF5

    if args.correctness_only:
        for dtype in TYPES:
            check_correctness(dtype, args.path, args.seed, fileFormat)
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
            fileFormat,
            comp_types,
        )
    elif args.only_read:
        time_ak_read(
            args.size, args.files_per_loc, args.trials, args.dtype, args.path, fileFormat, comp_types
        )
    else:
        time_ak_write(
            args.size,
            args.files_per_loc,
            args.trials,
            args.dtype,
            args.path,
            args.seed,
            fileFormat,
            comp_types,
        )
        time_ak_read(
            args.size, args.files_per_loc, args.trials, args.dtype, args.path, fileFormat, comp_types
        )
        remove_files(args.path)

    sys.exit(0)
