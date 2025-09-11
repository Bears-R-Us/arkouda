#!/usr/bin/env python3

"""
Usage: time python3 server_util/test/parallel_start_test.py -d `pwd`/test
"""

import optparse
import os
import os.path
import subprocess
import sys


SLOTS = 4


def process_dir(test_dir):
    """
    Process a directory containing Chapel unit tests

    Parameters
    ----------
    test_dir : Directory containing Chapel unit-test files

    Returns
    -------
    filtered view of files that are viable unit test files
    """
    utests = []
    skips = []
    if os.path.isdir(test_dir):
        for item in [i for i in os.listdir(test_dir)]:
            if item.lower().endswith("chpl") and "test" in item.lower():
                utests.append(item)
            elif item.lower().endswith(".notest"):
                skips.append(item[:-7] + ".chpl")  # Set up removals
            else:
                pass
        return filter(lambda v: v not in skips, utests)
    elif test_dir.lower().endswith(".chpl") and "test" in test_dir.lower():  # file case
        utests.append(test_dir)
        return utests
    else:
        raise Exception(f"Unknown Chapel unit tests file: {test_dir}")


def subprocess_start_test(fn):
    print(f"Calling subprocess on {fn}")
    cmd = ["start_test", fn]
    return subprocess.run(capture_output=True, args=cmd)


def successful_completed_process(p: subprocess.CompletedProcess):
    try:
        p.check_returncode()
        return True
    except subprocess.CalledProcessError:
        # print(f"{p.stderr}")
        return False


def parallel_test(unit_tests):
    import queue
    import threading

    q = queue.Queue()

    results = {}  # used to store the results
    for test in unit_tests:
        q.put(test)

    def worker():
        while True:
            unit_test = q.get()
            if unit_test is None:  # Reached Sentinel value, we're done
                q.task_done()
                return
            sp = subprocess_start_test(unit_test)
            if not successful_completed_process(sp):
                fn = sp.args[-1]
                results[fn] = "failed"
            q.task_done()

    threads = [threading.Thread(target=worker) for _i in range(SLOTS)]
    for thread in threads:
        thread.start()
        q.put(None)  # Sentinel value for each thread to know we're done

    q.join()
    return results


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-f", "--file", dest="filename", help="Run single test file", metavar="FILE")
    parser.add_option("-d", "--dir", dest="dir", help="Directory to look for test files")
    parser.add_option("-t", "--threads", dest="num_threads", help="Number of worker threads, default 4")

    (options, args) = parser.parse_args()
    if options.num_threads:
        SLOTS = int(options.num_threads)

    if options.dir and options.filename:
        raise Exception("Please specify only one of -f/--file or -d/--dir, not both")

    if options.dir or options.filename:
        tests = process_dir(options.dir if options.dir else options.filename)
        if options.dir:
            tests = [f"{options.dir}/{t}" for t in tests]  # Set up relative path to Unit tests
        print(f"Number of unit tests: {len(tests)}")
        print(f"Number of threads: {SLOTS}")
        print("Starting...")
        test_results = parallel_test(tests)
        print("Done")
        if len(test_results) > 0:
            print(f"{'-' * 40}")
            [print(f"Test {k} {v}") for k, v in test_results.items()]
            sys.exit(1)
    else:
        raise Exception("Please specify one of the options -f/--file or -d/--dir")
