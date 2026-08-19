#!/usr/bin/env python3

import argparse
import os
import time

from glob import glob
from math import ceil

import numpy as np
import pandas as pd

# import dask
# import dask.dataframe as dd
# from dask.distributed import Client
# from dask_jobqueue import PBSCluster

import arkouda as ak

from server_util.test.server_test_util import get_default_temp_directory


CHUNK_SIZE = 1_000_000
STD = 10.0
FILE_PATTERN = "measurements-*.parquet"
LOOKUP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources",
    "1trc-testing",
    "lookup.csv",
)


def load_lookup(path=LOOKUP_PATH):
    """Load the table of station names and their mean temperatures."""
    # generate_chunk maps positional station ids, so the index must stay a RangeIndex
    return pd.read_csv(path)


def generate_chunk(partition_idx, chunksize, std, lookup_df, out_dir="."):
    """Generate some sample data based on the lookup table."""
    rng = np.random.default_rng(partition_idx)  # Deterministic data generation
    df = pd.DataFrame(
        {
            # Choose a random station from the lookup table for each row in our output
            "station": rng.integers(0, len(lookup_df) - 1, int(chunksize)),
            # Generate a normal distibution around zero for each row in our output
            # Because the std is the same for every station we can adjust the mean for each row afterwards
            "measure": rng.normal(0, std, int(chunksize)),
        }
    )

    # Offset each measurement by the station's mean value
    df.measure += df.station.map(lookup_df.mean_temp)
    # Round the temprature to one decimal place
    df.measure = df.measure.round(decimals=1)
    # Convert the station index to the station name
    df.station = df.station.map(lookup_df.station)

    # Save this chunk to the output file
    filename = os.path.join(out_dir, f"measurements-{partition_idx}.parquet")
    df.to_parquet(filename, engine="pyarrow")


def generate_data(size, out_dir, chunksize=CHUNK_SIZE):
    """Write ``size`` rows of measurements to parquet files and return their directory."""
    os.makedirs(out_dir, exist_ok=True)
    lookup_df = load_lookup()
    for i in range(ceil(size / chunksize)):
        generate_chunk(i, chunksize, STD, lookup_df, out_dir)
    return out_dir


def measurement_files(data):
    files = sorted(glob(os.path.join(data, FILE_PATTERN)))
    if not files:
        raise ValueError(f"No files matching {FILE_PATTERN} found in {data}")
    return files


def dataset_bytes(files):
    """On-disk size of the dataset, used as the rate denominator."""
    return sum(os.path.getsize(f) for f in files)


def remove_files(path):
    for f in glob(os.path.join(path, FILE_PATTERN)):
        os.remove(f)


# def start_dask_cluster(args, num_jobs):
#     dask.config.set(
#         {
#             "distributed.scheduler.worker-ttl": "1h",
#             "distributed.comm.timeouts.connect": "120s",
#             "distributed.comm.timeouts.tcp": "120s",
#             "distributed.worker.memory.target": 0.6,
#             "distributed.worker.memory.spill": 0.7,
#             "distributed.worker.memory.pause": 0.85,
#             "distributed.worker.memory.terminate": False,
#             "temporary-directory": args.dask_scratch,
#         }
#     )
#
#     cluster_args = {
#         "cores": args.dask_cores,
#         "memory": args.dask_memory,
#         "walltime": args.dask_walltime,
#         "local_directory": args.dask_scratch,
#     }
#     if args.dask_queue:
#         cluster_args["queue"] = args.dask_queue
#     if args.dask_account:
#         cluster_args["account"] = args.dask_account
#
#     cluster = PBSCluster(**cluster_args)
#     client = Client(cluster)
#     print("scaling dask to {} PBS jobs".format(num_jobs))
#     cluster.scale(jobs=num_jobs)
#     client.wait_for_workers(n_workers=num_jobs)
#     return client, cluster


# def time_dask_1trc(trials, file_paths, totalbytes):
#     """Time the reference dask implementation of the challenge."""
#     print(">>> dask 1trc")
#
#     timings = []
#     result = None
#     for _ in range(trials):
#         start = time.time()
#         df = dd.read_parquet(file_paths, dtype_backend="pyarrow")
#         # split_out=1 forces a tree reduction
#         result = df.groupby("station").agg(["min", "max", "mean"], split_out=1).compute()
#         result = result.sort_values("station")
#         end = time.time()
#         timings.append(end - start)
#
#     tavg = sum(timings) / trials
#     print("dask Average time = {:.4f} sec".format(tavg))
#     print("dask Average rate = {:.4f} GiB/sec".format(totalbytes / tavg / 2**30))
#     print(result.head())
#     return tavg


def materialize_result_df(station_keys, mins, maxs, means):
    result_df = pd.DataFrame(
        {
            ("measure", "min"): mins.to_ndarray(),
            ("measure", "max"): maxs.to_ndarray(),
            ("measure", "mean"): means.to_ndarray(),
        },
        index=pd.Index(station_keys.to_ndarray(), name="station"),
    )
    return result_df


def time_ak_1trc(trials, file_paths, totalbytes):
    """Time the arkouda implementation of the challenge."""
    print(">>> arkouda 1trc")
    cfg = ak.get_config()
    print(
        "numLocales = {}, numNodes = {}, files = {:,}".format(
            cfg["numLocales"], cfg["numNodes"], len(file_paths)
        )
    )

    timings = []
    result = None
    for _ in range(trials):
        start = time.time()
        columns = ak.read(file_paths)
        stations = columns["station"]
        measures = columns["measure"]

        order = ak.argsort(stations)
        stations = stations[order]
        measures = measures[order]

        grouped = ak.GroupBy(stations, assume_sorted=True)
        station_keys, mins = grouped.min(measures)
        _, maxs = grouped.max(measures)
        _, means = grouped.mean(measures)
        end = time.time()

        timings.append(end - start)
        result = (station_keys, mins, maxs, means)
        # Release the per-trial server arrays outside the timed region
        del columns, stations, measures, order, grouped

    tavg = sum(timings) / trials
    print("arkouda Average time = {:.4f} sec".format(tavg))
    print("arkouda Average rate = {:.4f} GiB/sec".format(totalbytes / tavg / 2**30))

    # Pulling the results back to the client is not part of the measured time
    print(materialize_result_df(*result).head())
    return tavg


def create_parser():
    parser = argparse.ArgumentParser(description="Measure performance of the 1 trillion row challenge.")
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument("-n", "--size", type=int, default=10**8, help="Number of rows to compute with")
    parser.add_argument(
        "-t", "--trials", type=int, default=6, help="Number of times to run the benchmark"
    )
    parser.add_argument(
        "-d",
        "--data",
        required=False,
        type=str,
        help="Optional dataset directory to use, otherwise a dataset will be generated",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=os.path.join(get_default_temp_directory(), "1trc-test"),
        help="Target path for the generated dataset",
    )
    # parser.add_argument("--dask-cores", type=int, default=32, help="Cores per PBS job")
    # parser.add_argument("--dask-memory", default="400GB", help="Memory per PBS job")
    # parser.add_argument("--dask-walltime", default="5-00:00:00", help="Walltime per PBS job")
    # parser.add_argument("--dask-queue", default="", help="PBS queue to submit to")
    # parser.add_argument("--dask-account", default="", help="PBS account to charge")
    # parser.add_argument(
    #     "--dask-scratch",
    #     default=os.path.join(get_default_temp_directory(), "dask-scratch"),
    #     help="Worker scratch directory",
    # )
    return parser


if __name__ == "__main__":
    import sys

    args = create_parser().parse_args()
    setattr(ak, "verbose", False)
    ak.connect(args.hostname, args.port)

    # Use the supplied dataset if there is one, otherwise generate it
    data = args.data if args.data else generate_data(args.size, args.path, args.chunk_size)

    file_paths = measurement_files(data)
    totalbytes = dataset_bytes(file_paths)

    print("number of trials = ", args.trials)
    print("number of rows = ", args.size)

    # # Give dask the same number of jobs as the server has nodes so the two series line up
    # num_nodes = ak.get_config()["numNodes"]
    # dask_client, dask_cluster = start_dask_cluster(args, num_nodes)
    # try:
    #     dask_time = time_dask_1trc(args.trials, file_paths, totalbytes)
    # finally:
    #     dask_client.close()
    #     dask_cluster.close()

    arkouda_time = time_ak_1trc(args.trials, file_paths, totalbytes)

    # if dask_time and arkouda_time:
    #     print("arkouda/dask ratio = {:.2f}x".format(arkouda_time / dask_time))

    if not args.data:
        remove_files(args.path)

    sys.exit(0)
