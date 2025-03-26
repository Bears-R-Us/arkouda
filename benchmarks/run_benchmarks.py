#!/usr/bin/env python3
"""
This is a driver script to automatically run the Arkouda benchmarks in this
directory and optionally graph the results. Graphing requires that $CHPL_HOME
points to a valid Chapel directory. This will start and stop the Arkouda server
automatically.
"""

import argparse
import logging
import os
import subprocess
import sys

from server_util.test.server_test_util import (
    get_arkouda_numlocales,
    run_client,
    start_arkouda_server,
    stop_arkouda_server,
)


benchmark_dir = os.path.dirname(__file__)
util_dir = os.path.join(benchmark_dir, "..", "server_util", "test")
sys.path.insert(0, os.path.abspath(util_dir))

logging.basicConfig(level=logging.INFO)

BENCHMARKS = [
    "stream",
    "argsort",
    "coargsort",
    "groupby",
    "aggregate",
    "gather",
    "scatter",
    "reduce",
    "in1d",
    "scan",
    "noop",
    "setops",
    "array_create",
    "array_transfer",
    "IO",
    "csvIO",
    "small-str-groupby",
    "str-argsort",
    "str-coargsort",
    "str-groupby",
    "str-gather",
    "str-in1d",
    "substring_search",
    "split",
    "sort-cases",
    "multiIO",
    "str-locality",
    "dataframe",
    "encode",
    "bigint_conversion",
    "bigint_stream",
    "bigint_bitwise_binops",
    "bigint_groupby",
    "bigint_array_transfer",
]

if os.getenv("ARKOUDA_SERVER_PARQUET_SUPPORT"):
    BENCHMARKS.append("parquetIO")
    BENCHMARKS.append("parquetMultiIO")


def get_chpl_util_dir():
    """Get the Chapel directory that contains graph generation utilities."""
    CHPL_HOME = subprocess.check_output(["chpl", "--print-chpl-home"]).decode().strip()
    if not CHPL_HOME:
        logging.error("$CHPL_HOME not set")
        sys.exit(1)
    chpl_util_dir = os.path.join(CHPL_HOME, "util", "test")
    if not os.path.isdir(chpl_util_dir):
        logging.error("{} does not exist".format(chpl_util_dir))
        sys.exit(1)
    return chpl_util_dir


def _my_start_server(args):
    start_arkouda_server(
        numlocales=args.num_locales,
        port=args.server_port,
        server_args=args.server_args,
        within_slurm_alloc=bool(args.within_slurm_alloc),
    )


def add_to_dat(benchmark, output, dat_dir, graph_infra):
    """
    Run computePerfStats to take output from a benchmark and create/append to a
    .dat file that contains performance keys. The performance keys come from
    `graph_infra/<benchmark>.perfkeys` if it exists, otherwise a default
    `graph_infra/perfkeys` is used.
    """
    computePerfStats = os.path.join(get_chpl_util_dir(), "computePerfStats")

    perfkeys = os.path.join(graph_infra, "{}.perfkeys".format(benchmark))
    if not os.path.exists(perfkeys):
        perfkeys = os.path.join(graph_infra, "perfkeys")

    benchmark_out = "{}.exec.out.tmp".format(benchmark)
    with open(benchmark_out, "w") as f:
        f.write(output)
    subprocess.check_output([computePerfStats, benchmark, dat_dir, perfkeys, benchmark_out])
    os.remove(benchmark_out)


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

    # TODO support alias for a larger default N
    # parser.add_argument('--large', default=False, action='store_true',
    # help='Run a larger problem size')

    parser.add_argument(
        "-nl",
        "--num-locales",
        "--numLocales",
        default=get_arkouda_numlocales(),
        help="Number of locales to use for the server",
    )
    parser.add_argument("-sp", "--server-port", default="5555", help="Port number to use for the server")
    parser.add_argument("--server-args", action="append", help="Additional server arguments")
    parser.add_argument("--numtrials", default=1, type=int, help="Number of trials to run")
    parser.add_argument(
        "benchmarks",
        nargs="*",
        help="Basename of benchmarks to run with extension stripped",
    )
    parser.add_argument(
        "--save-data",
        default=False,
        action="store_true",
        help="Save performance data to output files, requires $CHPL_HOME",
    )
    parser.add_argument(
        "--gen-graphs",
        default=False,
        action="store_true",
        help="Generate graphs, requires $CHPL_HOME",
    )
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
    parser.add_argument(
        "--isolated",
        default=False,
        help="run each benchmark in its own server instance",
    )
    parser.add_argument(
        "--within-slurm-alloc",
        default=False,
        help="whether this script was launched from within a slurm allocation "
        + "(for use with --isolated only)",
    )
    return parser


def main():
    parser = create_parser()
    args, client_args = parser.parse_known_args()
    args.graph_dir = args.graph_dir or os.path.join(args.dat_dir, "html")
    config_dat_dir = os.path.join(args.dat_dir, args.description)
    run_isolated = bool(args.isolated)

    if args.save_data or args.gen_graphs:
        os.makedirs(config_dat_dir, exist_ok=True)

    if not run_isolated:
        _my_start_server(args)

    args.benchmarks = args.benchmarks or BENCHMARKS
    for benchmark in args.benchmarks:
        if run_isolated:
            _my_start_server(args)

        for trial in range(args.numtrials):
            benchmark_py = os.path.join(benchmark_dir, "{}.py".format(benchmark))
            out = run_client(benchmark_py, client_args)
            if args.save_data or args.gen_graphs:
                add_to_dat(benchmark, out, config_dat_dir, args.graph_infra)
            print(out)

        if run_isolated:
            stop_arkouda_server()

    if not run_isolated:
        stop_arkouda_server()

    if args.save_data or args.gen_graphs:
        comp_file = os.getenv("ARKOUDA_PRINT_PASSES_FILE", "")
        if os.path.isfile(comp_file):
            with open(comp_file, "r") as f:
                out = f.read()
            add_to_dat("comp-time", out, config_dat_dir, args.graph_infra)
        emitted_code_file = os.getenv("ARKOUDA_EMITTED_CODE_SIZE_FILE", "")
        if os.path.isfile(emitted_code_file):
            with open(emitted_code_file, "r") as f:
                out = f.read()
            add_to_dat("emitted-code-size", out, config_dat_dir, args.graph_infra)
        if args.gen_graphs:
            generate_graphs(args)


if __name__ == "__main__":
    main()
