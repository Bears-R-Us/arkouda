# !/usr/bin/env python3
# test data located at  https://csr.lanl.gov/data/netflow.html
# field names located at https://csr.lanl.gov/data/2017.html

import os
import sys
import time
from glob import glob

from base_test import ArkoudaTest
from context import arkouda as ak


class ReadAllTest(ArkoudaTest):
    def setUp(self):

        """
        Invokes the parent setUp method to start the arkouda_server and also
        sets the test_data_url used to lookup test files.sets

        :return: None
        :raise: AssertionError if the TEST_DATA_URL has not been set
        """
        ArkoudaTest.setUp(self)
        self.test_data_url = os.getenv("TEST_DATA_URL")
        assert self.test_data_url, "The TEST_DATA_URL env variable must be set"

    def testReadAll(self):
        ak.verbose = False  # client verbose Flag
        cwd = os.getcwd()
        allfiles = glob(cwd + "/../converter/netflow_day-*.hdf")
        print(allfiles)
        start = time.time()
        dictionary1 = ak.read(allfiles, iterative=True)
        end = time.time()
        t1 = end - start
        print("read(iterative=True) seconds: %.3f" % (t1))
        for key, value in dictionary1.items():
            print(key, type(value), value, len(value))

        start = time.time()
        dictionary2 = ak.read(allfiles)
        end = time.time()
        t2 = end - start
        print("read() seconds: %.3f" % (t2))
        for key, value in dictionary2.items():
            print(key, type(value), value, len(value))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: {} <hostname> <port> <HDF5_filenames>".format(sys.argv[0]))
        sys.exit()
    ak.connect(sys.argv[1], sys.argv[2])
    ak.verbose = False  # client verbose Flag
    cwd = os.getcwd()
    allfiles = glob(cwd + "/../converter/netflow_day-*.hdf")
    if len(sys.argv) > 3:
        allfiles = sys.argv[3:]

    start = time.time()
    dictionary1 = ak.read(allfiles, iterative=True)
    end = time.time()
    t1 = end - start
    print("read(iterative=True) seconds: %.3f" % (t1))
    for key, value in dictionary1.items():
        print(key, type(value), value, len(value))

    start = time.time()
    dictionary2 = ak.read(allfiles)
    end = time.time()
    t2 = end - start
    print("read() seconds: %.3f" % (t2))
    for key, value in dictionary2.items():
        print(key, type(value), value, len(value))

    ak.disconnect()
