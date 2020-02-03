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
if len(sys.argv) > 2:
    allfiles = sys.argv[3:]

#read all
#print(f"df = ak.read_all({allfiles},['Time', 'Duration', 'Protocol'])")
#pdArrayDictionary = ak.read_all(allfiles, ['Time', 'Duration', 'Protocol','SrcDevice'],True)
#pdArrayDictionary = ak.read_all(allfiles, ['Time', 'SrcDevice'],True)
#pdArrayDictionary = ak.read_all(allfiles, ['Time', 'SrcDevice'],True)
#pdArray = ak.read_all(allfiles, 'start', True)
start = time.time()
pdArrayDictionary1 = ak.read_all(allfiles, iterative=True)
end = time.time()
t1 = end - start
start = time.time()
pdArrayDictionary2 = ak.read_all(allfiles)
end = time.time()
t2 = end - start

print("read_all(iterative=True) seconds: %.3f \nread_all(iterative=False) seconds: %.3f" % (t1, t2))

#for key, value in pdArrayDictionary2.items():
#    print(key,type(value),value)

ak.disconnect()
