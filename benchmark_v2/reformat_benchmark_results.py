#!/usr/bin/env python3
"""
This is a driver script to automatically run the Arkouda benchmarks in this
directory and optionally graph the results. Graphing requires that $CHPL_HOME
points to a valid Chapel directory. This will start and stop the Arkouda server
automatically.
"""
import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Union

benchmark_dir = os.path.dirname(__file__)
util_dir = os.path.join(benchmark_dir, "..", "server_util", "test")
sys.path.insert(0, os.path.abspath(util_dir))

logging.basicConfig(level=logging.INFO)

BENCHMARKS = [
    # "stream",
    "argsort",
    "coargsort",
    # "groupby",
    "aggregate",
    # "gather",
    # "scatter",
    # "reduce",
    # "in1d",
    # "scan",
    # "noop",
    # "setops",
    # "array_create",
    # "array_transfer",
    # "IO",
    # "csvIO",
    # "small-str-groupby",
    # "str-argsort",
    # "str-coargsort",
    # "str-groupby",
    # "str-gather",
    # "str-in1d",
    # "substring_search",
    # "split",
    # "sort-cases",
    # "multiIO",
    # "str-locality",
    # "dataframe",
    # "encode",
    # "bigint_conversion",
    # "bigint_stream",
    # "bigint_bitwise_binops",
    # "bigint_groupby",
    # "bigint_array_transfer",
]

if os.getenv("ARKOUDA_SERVER_PARQUET_SUPPORT"):
    BENCHMARKS.append("parquetIO")
    BENCHMARKS.append("parquetMultiIO")


def get_chpl_util_dir():
    """Get the Chapel directory that contains graph generation utilities."""
    CHPL_HOME = os.getenv("CHPL_HOME")
    if not CHPL_HOME:
        logging.error("$CHPL_HOME not set")
        sys.exit(1)
    chpl_util_dir = os.path.join(CHPL_HOME, "util", "test")
    if not os.path.isdir(chpl_util_dir):
        logging.error("{} does not exist".format(chpl_util_dir))
        sys.exit(1)
    return chpl_util_dir


def generate_graphs(args):
    """
    Generate graphs using the existing .dat files and graph infrastructure.
    """
    genGraphs = os.path.join(get_chpl_util_dir(), "genGraphs")
    cmd = [
        genGraphs,
        "--perfdir",
        args.dat_dir,
        "--outdir",
        args.graph_dir,
        "--graphlist",
        os.path.join(args.graph_infra, "GRAPHLIST"),
        "--testdir",
        args.graph_infra,
        "--alttitle",
        "Arkouda Performance Graphs",
    ]

    if args.platform_name:
        cmd += ["--name", args.platform_name]
    if args.configs:
        cmd += ["--configs", args.configs]
    if args.start_date:
        cmd += ["--startdate", args.start_date]
    if args.annotations:
        cmd += ["--annotate", args.annotations]

    subprocess.check_output(cmd)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dat-dir",
        default=os.path.join(benchmark_dir, "datdir"),
        help="Directory with .dat files stored",
    )
    parser.add_argument("--graph-dir", help="Directory to place generated graphs")
    parser.add_argument(
        "--graph-infra",
        default=os.path.join(benchmark_dir, "graph_infra"),
        help="Directory containing graph infrastructure",
    )
    parser.add_argument("--platform-name", default="", help="Test platform name")
    parser.add_argument("--description", default="", help="Description of this configuration")
    parser.add_argument("--annotations", default="", help="File containing annotations")
    parser.add_argument("--configs", help="comma seperate list of configurations")
    parser.add_argument("--start-date", help="graph start date")
    parser.add_argument("--benchmark-data", help="the benchnmark output data in json format.")

    return parser


def get_header_dict(directory_path):
    """
    Returns a dictionary where the keys are the benchmark names and the
    values are the lists of header fields for the associated .dat files.
    The headers are read in from .perfkeys files in directory_path, when possible,
    otherwise the default values are used.  The "# Date" field is also appended to each header.

    """
    headers = get_header_fields_from_directory(directory_path)
    for benchmark_name in BENCHMARKS:
        if benchmark_name not in headers.keys():
            headers[benchmark_name] = ["Average time =", "Average rate ="]

    for key in headers.keys():
        headers[key].insert(0, "# Date")

    return headers


# This algorithm for reading files into a dictionary was generated with assistance from Perplexity AI (2024).
def get_header_fields_from_directory(directory_path):
    """
    Returns a dictionary where the keys are the benchmark names and the
    values are the lists of header fields for the associated .dat files.
    Only retrieves header fields in the .perfkeys files in the graph_infra directory,
    and does not include the default fields.
    """
    # Dictionary to store file names and their contents
    file_contents = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if it's a file (not a directory)
        if (".perfkeys" in file_path) and os.path.isfile(file_path):
            try:
                # Open and read the file
                with open(file_path, "r", encoding="utf-8") as file:
                    # Read all lines and store them in a list
                    lines = file.readlines()

                    # Strip newline characters from each line
                    lines = [line.strip() for line in lines]

                    # Add the file name and its contents to the dictionary
                    key = re.search(r"([\w_]+).perfkeys", filename)[1]
                    file_contents[key] = lines
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")

    return file_contents


def get_nested_value(data: dict, keys: list):
    """Look up a value in a dictionary using a list of keys."""
    for key in keys:
        if isinstance(data, dict):
            # Replace data with data[key], using {} as the default option.
            data = data.get(key, {})
        else:
            return None
    return data if data != {} else None


def get_value(field: str, benchmark_name: str, field_lookup_map: dict, benchmark_data):
    """get the value of a field in a benchmark using the field_lookup_map and the benchmark_data in pytest json format."""
    regex_str = None
    if (
        field_lookup_map.get(benchmark_name).get(field) is not None
        and isinstance(field_lookup_map.get(benchmark_name).get(field).get("lookup_regex"), str)
        and field_lookup_map.get(benchmark_name).get(field).get("lookup_regex") != ""
    ):
        regex_str = field_lookup_map.get(benchmark_name).get(field).get("lookup_regex")

    lookup_path = None
    if field_lookup_map.get(benchmark_name).get(field) is not None and isinstance(
        field_lookup_map.get(benchmark_name).get(field).get("lookup_path"), list
    ):
        lookup_path = field_lookup_map.get(benchmark_name).get(field).get("lookup_path")

    if field == "# Date":
        date_str = benchmark_data["datetime"]
        return datetime.fromisoformat(date_str).strftime("%m/%d/%y")
    elif regex_str is not None and regex_str != "":
        return compute_average(regex_str, lookup_path, benchmark_data)
    elif benchmark_name in field_lookup_map.keys() and field in field_lookup_map[benchmark_name].keys():
        group = field_lookup_map[benchmark_name][field]["group"]
        name = field_lookup_map[benchmark_name][field]["name"]
        lookup_path = field_lookup_map[benchmark_name][field]["lookup_path"]

        for benchmark in benchmark_data["benchmarks"]:
            if (benchmark["group"] == group) and (benchmark["name"] == name):
                value = get_nested_value(benchmark, lookup_path)
                return get_float_value(value)

    print(f"Could not get value for {field} in {benchmark_name} data.")
    return -1.0


def compute_average(benchmark_name_regex: str, keys: list, benchmark_data):
    """Compute the average value of a statistic, using a regex on the benchmark name to determine which values to use."""
    sum = 0.0
    N = 0
    for benchmark in benchmark_data["benchmarks"]:
        if re.match(benchmark_name_regex, benchmark["name"]):
            value = get_float_value(get_nested_value(benchmark, keys))
            sum += value
            N += 1
    if N > 0:
        return sum / N
    else:
        print(f"Could not compute average over {benchmark_name_regex}.")
        return -1.0


def get_float_value(value: Union[float, str]):
    if isinstance(value, str):
        # extract float value:
        return float(re.search(r"[\d\.]+", value)[0])
    elif isinstance(value, float):
        return value
    else:
        raise TypeError("In get_float_value, value must be a float or string.")


def gen_lookup_map(write=False, out_file="field_lookup_map.json"):
    """Temporarily use a script to generate the lookup dictionary and save to file when write=True."""
    field_lookup_map = {}
    for benchmark_name in BENCHMARKS:
        field_lookup_map[benchmark_name] = {}

        field_lookup_map[benchmark_name]["Average rate ="] = get_lookup_dict(
            name="bench_" + benchmark_name,
            benchmark_name=benchmark_name,
            lookup_path=[
                "extra_info",
                "transfer_rate",
            ],
            lookup_regex="bench_" + benchmark_name + r"\[[\w\d]*\]",
        )

        field_lookup_map[benchmark_name]["Average time ="] = get_lookup_dict(
            name="bench_" + benchmark_name,
            benchmark_name=benchmark_name,
            lookup_path=[
                "stats",
                "mean",
            ],
            lookup_regex="bench_" + benchmark_name + r"\[[\w\d]*\]",
        )

    for op in [
        "prod",
        "sum",
        "mean",
        "min",
        "max",
        "argmin",
        "argmax",
        "any",
        "all",
        "xor",
        "and",
        "or",
        "nunique",
    ]:

        field_lookup_map["aggregate"][f"Aggregate {op} Average rate ="] = get_lookup_dict(
            group="GroupBy.aggregate",
            name=f"bench_aggregate[{op}]",
            benchmark_name="aggregate",
            lookup_path=[
                "extra_info",
                "transfer_rate",
            ],
        )

        field_lookup_map["aggregate"][f"Aggregate {op} Average time ="] = get_lookup_dict(
            group="GroupBy.aggregate",
            name=f"bench_aggregate[{op}]",
            benchmark_name="aggregate",
            lookup_path=[
                "stats",
                "mean",
            ],
        )

    for num in [1, 2, 8, 16]:

        field_lookup_map["coargsort"][f"{num}-array Average rate ="] = get_lookup_dict(
            group="Arkouda_CoArgSort",
            benchmark_name="coargsort",
            lookup_path=["extra_info", "transfer_rate"],
            lookup_regex=f"bench_coargsort\\[[\\w\\d]*-{num}\\]",
        )

        field_lookup_map["coargsort"][f"{num}-array Average time ="] = get_lookup_dict(
            group="Arkouda_CoArgSort",
            benchmark_name="coargsort",
            lookup_path=[
                "stats",
                "mean",
            ],
            lookup_regex=f"bench_coargsort\\[[\\w\\d]*-{num}\\]",
        )

    if write:
        with open(out_file, "w") as fp:
            json.dump(field_lookup_map, fp)

    return field_lookup_map


def get_lookup_dict(group="", name="", benchmark_name="", lookup_path=[], lookup_regex=""):
    """Populate the lookup dictionary fields and return a dictionary."""
    ret_dict = {
        "group": group,
        "name": name,
        "benchmark_name": benchmark_name,
        "lookup_path": lookup_path,
        "lookup_regex": lookup_regex,
    }
    return ret_dict


# ./benchmark_v2/reformat_benchmark_results.py
def main():
    parser = create_parser()
    args, client_args = parser.parse_known_args()
    args.graph_dir = args.graph_dir or os.path.join(args.dat_dir, "html")
    configs_dir = os.path.join(args.dat_dir, "configs")
    benchmark_data_path = args.benchmark_data

    os.makedirs(configs_dir, exist_ok=True)

    lookup_map_path = configs_dir + "/field_lookup_map.json"

    #   TODO: remove gen_lookup_map
    gen_lookup_map(True, lookup_map_path)

    with open(lookup_map_path, "r") as file:
        field_lookup_map = json.load(file)

    headers = get_header_dict(args.graph_infra)

    #   Load benchmark data
    with open(benchmark_data_path, "r") as file:
        benchmark_data = json.load(file)

    #   Convert benchmark data to output rows, in dictionary format with benchmark names as the keys.
    out_data = {}

    for benchmark_name in BENCHMARKS:
        if benchmark_name not in headers.keys():
            print(f"Could could not find headers for {benchmark_name}.")
        else:
            header = headers[benchmark_name]
            row = [
                get_value(field, benchmark_name, field_lookup_map, benchmark_data)
                for field in header
            ]

            if benchmark_name in out_data.keys() and isinstance(out_data[benchmark_name], list):
                out_data[benchmark_name].append(row)
            else:
                out_data[benchmark_name] = [row]

    #   Write the outputs to .dat files
    for benchmark_name in BENCHMARKS:
        if benchmark_name not in out_data.keys():
            print(f"Could not generate {benchmark_name}.dat\nskipping....")
            continue

        data_file = args.dat_dir + f"/{benchmark_name}.dat"
        header = headers[benchmark_name]

        if not os.path.exists(data_file):
            with open(data_file, "a", newline="") as file:
                writer = csv.writer(file, delimiter="\t")
                writer.writerow(header)

        if out_data[benchmark_name] is not None:
            with open(data_file, "a", newline="") as file:
                writer = csv.writer(file, delimiter="\t")
                writer.writerows(out_data[benchmark_name])

    generate_graphs(args)


if __name__ == "__main__":
    main()
