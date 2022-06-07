#!/usr/bin/env python3

import os
import sys


def import_local(path):
    if not os.path.exists(path):
        raise ImportError(f"{path} not found")
    importdir, filename = os.path.split(path)
    importname, ext = os.path.splitext(filename)
    if ext != ".py":
        raise ImportError(f"{path} must be a .py file")
    sys.path.append(importdir)
    return f"from {importname} import OPTIONS as CUSTOM"


if __name__ == "__main__":
    import argparse
    from multiprocessing import cpu_count

    import hdflow

    parser = argparse.ArgumentParser(
        description="Convert CSV files with numeric data to HDF5 files in parallel."
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory for HDF5 files (Default: current directory)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=cpu_count(),
        help="Number of worker processes to use (Default: all available cores)",
    )
    parser.add_argument("--extension", default=".hdf", help="Output file extension (Default: .hdf)")
    parser.add_argument(
        "--format",
        required=True,
        help="Name of netflow format defined in --formats-file argument",
    )
    parser.add_argument(
        "--formats-file",
        required=True,
        help="Python file specifying read_csv options, e.g. column names and dtypes",
    )
    parser.add_argument("filenames", nargs="+", help="Input files to convert")

    args = parser.parse_args()
    exec(import_local(args.formats_file))
    if args.format not in CUSTOM:
        raise ValueError(f"Netflow format not found. Detected formats: {set(CUSTOM.keys())}")
    hdflow.convert_files(args.filenames, args.outdir, args.extension, CUSTOM[args.format], args.jobs)
