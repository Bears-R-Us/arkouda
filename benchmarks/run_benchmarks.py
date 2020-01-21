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

benchmark_dir = os.path.dirname(__file__)
util_dir = os.path.join(benchmark_dir, '..', 'util', 'test')
sys.path.insert(0, os.path.abspath(util_dir))
from util import *

logging.basicConfig(level=logging.INFO)

BENCHMARKS = ['stream', 'argsort', 'gather', 'scatter', 'reduce', 'scan']

def get_chpl_util_dir():
    """ Get the Chapel directory that contains graph generation utilities. """
    CHPL_HOME = os.getenv('CHPL_HOME')
    chpl_util_dir = os.path.join(CHPL_HOME, 'util', 'test')
    if not CHPL_HOME or not os.path.isdir(chpl_util_dir):
        logging.error('$CHPL_HOME not set, or {} missing'.format(chpl_util_dir))
        sys.exit(1)
    return chpl_util_dir

def add_to_dat(benchmark, output, dat_dir, graph_infra):
    """
    Run computePerfStats to take output from a benchmark and create/append to a
    .dat file that contains performance keys. The performance keys come from
    `graph_infra/<benchmark>.perfkeys` if it exists, otherwise a default
    `graph_infra/perfkeys` is used.
    """
    computePerfStats = os.path.join(get_chpl_util_dir(), 'computePerfStats')

    perfkeys = os.path.join(graph_infra, '{}.perfkeys'.format(benchmark))
    if not os.path.exists(perfkeys):
        perfkeys = os.path.join(graph_infra, 'perfkeys')

    benchmark_out = '{}.exec.out.tmp'.format(benchmark)
    with open (benchmark_out, 'w') as f:
        f.write(output)
    subprocess.check_output([computePerfStats, benchmark, dat_dir, perfkeys, benchmark_out])
    os.remove(benchmark_out)

def generate_graphs(dat_dir, graph_dir, graph_infra, platform_name):
    """
    Generate graphs using the existing .dat files and graph infrastructure.
    """
    genGraphs = os.path.join(get_chpl_util_dir(), 'genGraphs')
    cmd = [genGraphs,
           '--perfdir', dat_dir,
           '--outdir', graph_dir,
           '--graphlist', os.path.join(graph_infra, 'GRAPHLIST'),
           '--testdir', graph_infra,
           '--alttitle', 'Arkouda Performance Graphs']

    if platform_name:
        cmd += ['--name', platform_name]

    subprocess.check_output(cmd)

def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    # TODO support passing through size/trials/type/whatever to benchmarks
    #parser.add_argument('-n', '--size', type=int, default=10**8, help='Problem size: length of arrays A and B')
    #parser.add_argument('-t', '--trials', type=int, default=6, help='Number of times to run the benchmark')
    #parser.add_argument('-d', '--dtype', default='int64', help='Dtype of arrays (int64 or float64)')
    #parser.add_argument('-r', '--randomize', default=False, action='store_true', help='Fill arrays with random values instead of ones')

    # TODO support running numpy variations
    #parser.add_argument('--numpy', default=False, action='store_true', help='Run the same operation in NumPy to compare performance.')

    # TODO support running correctness mode only
    # parser.add_argument('correctnss', default=False, action='store_true', help='Run correctness checks only')

    # TODO support alias for a larger default N
    #parser.add_argument('--large', default=False, action='store_true', help='Run a larger problem size')

    parser.add_argument('-nl', '--num-locales', default=get_arkouda_numlocales(), help='Number of locales to use for the server')
    parser.add_argument('--gen-graphs', default=False, action='store_true', help='Generate graphs, requires $CHPL_HOME')
    parser.add_argument('--dat-dir', default=os.path.join(benchmark_dir, 'datdir'), help='Directory with .dat files stored')
    parser.add_argument('--graph-dir', default=os.path.join(benchmark_dir, 'graphdir'), help='Directory to place generated graphs')
    parser.add_argument('--graph-infra', default=os.path.join(benchmark_dir, 'graph_infra'), help='Directory containing graph infrastructure')
    parser.add_argument('--platform-name', default=None, help='Test platform name')
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.gen_graphs:
        os.makedirs(args.dat_dir, exist_ok=True)

    start_arkouda_server(args.num_locales)

    for benchmark in BENCHMARKS:
        benchmark_py = os.path.join(benchmark_dir, '{}.py'.format(benchmark))
        out = run_client(benchmark_py)
        if args.gen_graphs:
            add_to_dat(benchmark, out, args.dat_dir, args.graph_infra)
        print(out)

    stop_arkouda_server()

    if args.gen_graphs:
        generate_graphs(args.dat_dir, args.graph_dir, args.graph_infra, args.platform_name)

if __name__ == '__main__':
    main()
