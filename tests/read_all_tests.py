# !/usr/bin/env python3
# test data located at  https://csr.lanl.gov/data/netflow.html
# field names located at https://csr.lanl.gov/data/2017.html

import arkouda as ak
import sys, os, h5py, time
from glob import glob

if len(sys.argv) < 3:
    print("Usage: {} <hostname> <port> <HDF5_filenames>".format(sys.argv[0]))
    sys.exit()
ak.connect(sys.argv[1], sys.argv[2])
ak.verbose = False    #client verbose Flag
cwd = os.getcwd()
allfiles = glob(cwd+'/../converter/netflow_day-*.hdf')
if len(sys.argv) > 3:
    allfiles = sys.argv[3:]

start = time.time()
dictionary1 = ak.read_all(allfiles, iterative=True)
end = time.time()
t1 = end - start
print("read_all(iterative=True) seconds: %.3f" % (t1))
for key, value in dictionary1.items():
    print(key,type(value),value,len(value))

start = time.time()
dictionary2 = ak.read_all(allfiles)
end = time.time()
t2 = end - start
print("read_all() seconds: %.3f" % (t2))
for key, value in dictionary2.items():
    print(key,type(value),value,len(value))

ak.disconnect()
