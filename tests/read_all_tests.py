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
allfiles = glob(cwd+'/../converter/netflow_day-1*.hdf')
if len(sys.argv) > 3:
    allfiles = sys.argv[3:]

start = time.time()
pdArrayDictionary1 = ak.read_all(allfiles, ['dstIP','start'], iterative=True)
end = time.time()
t1 = end - start
print("read_all(iterative=True) seconds: %.3f" % (t1))
for key, value in pdArrayDictionary1.items():
    print(key,type(value),value)

start = time.time()
pdArrayDictionary2 = ak.read_all(allfiles, ['dstIP','start'])
end = time.time()
t2 = end - start
print("read_all() seconds: %.3f" % (t2))
for key, value in pdArrayDictionary2.items():
    print(key,type(value),value)

ak.disconnect()
