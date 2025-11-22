import argparse
import os
import shutil
import time

import pandas as pd

import arkouda as ak

from server_util.test.server_test_util import get_default_temp_directory


str_length = 2
test_dir = ""
test_results = {
    "single-file": 0,
    "fixed-single": 0,
    "scaled-five": 0,
    "fixed-scaled-five": 0,
    "five": 0,
    "fixed-five": 0,
    "scaled-ten": 0,
    "fixed-scaled-ten": 0,
    "ten": 0,
    "fixed-ten": 0,
}

correctness_test = False


def generate_arr(num_files, scaling):
    if scaling:
        return ak.random_strings_uniform(str_length, str_length + 1, int(size / num_files))
    else:
        return ak.random_strings_uniform(str_length, str_length + 1, size)


def compare_arrs(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            print("FAIL!")
            print(a[i], "!=", b[i], " at ", i)


def write_files():
    # write 1 file with size
    a = generate_arr(1, False)
    a.to_parquet(test_dir + "single-file")
    # write 5 files with size/5
    for i in range(5):
        a = generate_arr(5, True)
        a.to_parquet(test_dir + "scaled-five" + str(i))
    # write 5 files with size
    for i in range(5):
        a = generate_arr(5, False)
        a.to_parquet(test_dir + "five" + str(i))
    # write 10 files with size/10
    for i in range(10):
        a = generate_arr(10, True)
        a.to_parquet(test_dir + "scaled-ten" + str(i))
    # write 10 files with size
    for i in range(10):
        a = generate_arr(10, False)
        a.to_parquet(test_dir + "ten" + str(i))


def read_files_fixed():
    start = time.time()
    a = ak.read(test_dir + "single-file*", fixed_len=str_length)
    stop = time.time()
    test_results["fixed-single"] += stop - start

    start = time.time()
    a = ak.read(test_dir + "scaled-five*", fixed_len=str_length)
    stop = time.time()
    test_results["fixed-scaled-five"] += stop - start

    start = time.time()
    a = ak.read(test_dir + "five*", fixed_len=str_length)
    stop = time.time()
    test_results["fixed-five"] += stop - start

    start = time.time()
    a = ak.read(test_dir + "scaled-ten*", fixed_len=str_length)
    stop = time.time()
    test_results["fixed-scaled-ten"] += stop - start

    start = time.time()
    a = ak.read(test_dir + "ten*", fixed_len=str_length)
    stop = time.time()
    test_results["fixed-ten"] += stop - start


def read_files():
    start = time.time()
    a = ak.read(test_dir + "single-file*")
    stop = time.time()
    test_results["single-file"] += stop - start

    start = time.time()
    a = ak.read(test_dir + "scaled-five*")
    stop = time.time()
    test_results["scaled-five"] += stop - start

    start = time.time()
    a = ak.read(test_dir + "five*")
    stop = time.time()
    test_results["five"] += stop - start

    start = time.time()
    a = ak.read(test_dir + "scaled-ten*")
    stop = time.time()
    test_results["scaled-ten"] += stop - start

    start = time.time()
    a = ak.read(test_dir + "ten*")
    stop = time.time()
    test_results["ten"] += stop - start


def delete_folder_contents(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")


def print_performance_table(test_results):
    data = [(test, time) for test, time in test_results.items()]
    df = pd.DataFrame(data, columns=["test", "sec"])
    df["sec"] = df["sec"].apply(lambda x: f"{x:.3f}")
    print(df.to_markdown(index=False))


def create_parser():
    parser = argparse.ArgumentParser(
        description="Measure performance of writing and reading random arrays from disk."
    )
    parser.add_argument("hostname", help="Hostname of arkouda server")
    parser.add_argument("port", type=int, help="Port of arkouda server")
    parser.add_argument(
        "-n", "--size", type=int, default=10**8, help="Problem size: length of array to write/read"
    )
    parser.add_argument(
        "-w",
        "--write",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--fixed",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=os.path.join(get_default_temp_directory(), "ak-io-test"),
        help="Target path for measuring read/write rates",
    )
    return parser


if __name__ == "__main__":
    import sys

    parser = create_parser()
    args = parser.parse_args()
    ak.connect(args.hostname, args.port)
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    test_dir = args.path
    size = args.size

    write_files()

    read_files_fixed()
    read_files()

    delete_folder_contents(test_dir)

    print_performance_table(test_results)
