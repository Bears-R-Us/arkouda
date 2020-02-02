# !/usr/bin/env python3
# test data located at  https://csr.lanl.gov/data/netflow.html
# field names located at https://csr.lanl.gov/data/2017.html

import arkouda as ak
import sys, os
import h5py
from glob import glob
import os

#print(os.path.realpath(__file__))
#print(cwd)
saveone = '/tmp/ak_save.hdf'
##print(saveone)
#h5py.File('/tmp/ak_save.hdf','w')
saveall = '/tmp/ak_save_all.hdf'
#h5py.File('/tmp/ak_save_all.hdf','w')

#print(saveone)

if len(sys.argv) < 4:
    print("Usage: {} <hostname> <port> <HDF5_filenames>".format(sys.argv[0]))
    sys.exit()
ak.connect(sys.argv[1], sys.argv[2])
ak.verbose = True

cwd = os.getcwd()


#read all
allfiles = sys.argv[3:]
#print(f"df = ak.read_all({allfiles},['Time', 'Duration', 'Protocol'])")
#pdArrayDictionary = ak.read_all(allfiles, ['Time', 'Duration', 'Protocol','SrcDevice'],True)
#pdArrayDictionary = ak.read_all(allfiles, ['Time', 'SrcDevice'],True)
#pdArrayDictionary = ak.read_all(allfiles, ['Time', 'SrcDevice'],True)
#pdArray = ak.read_all(allfiles, 'start', True)
pdArrayDictionary = ak.read_all(glob(cwd+'/../../../bigdata/hdf/netflow_day-1*.h5'),['start', 'duration'], iterative=False)
#pdArrayDictionary = ak.read_all(glob(cwd+'/../../../bigdata/hdf/netflow_day-1*.h5'), iterative=False)


#pdArrayDictionary = ak.read_all(allfiles,None,True)
#print(f"ak.save_all(df, {saveall})")
#ak.save_all(df, saveall)
#print(f"newdf = ak.load_all({saveall})")
#newdf = ak.load_all(saveall)
#print(newdf)
#os.system('rm -rf ../tmp/ak_save*')
ak.disconnect()
