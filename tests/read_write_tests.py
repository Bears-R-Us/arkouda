#!/usr/bin/env python3

import os
import sys

import arkouda as ak

saveone = "/tmp/ak_save.hdf"
saveall = "/tmp/ak_save_all.hdf"

if len(sys.argv) < 4:
    print("Usage: {} <hostname> <port> <HDF5_filenames>".format(sys.argv[0]))
    sys.exit()
ak.connect(sys.argv[1], sys.argv[2])
onefile = sys.argv[3]
print(ak.ls(onefile))
allfiles = sys.argv[3:]
print(f"srcIP = ak.read({onefile},'srcIP')")
srcIP = ak.read(onefile, "srcIP")
print(f"srcIP.save({saveone}, 'srcIP')")
srcIP.save(saveone, "srcIP")
print(f"srcIP2 = ak.load({saveone}, 'srcIP')")
srcIP2 = ak.load(saveone, "srcIP")
assert (srcIP == srcIP2).all()
del srcIP
del srcIP2
print(f"df = ak.read(['srcPort', 'proto', 'packets'], {allfiles})")
df = ak.read(["srcPort", "proto", "packets"], allfiles)
print(f"ak.save_all(df, {saveall})")
ak.save_all(df, saveall)
print(f"newdf = ak.load_all({saveall})")
newdf = ak.load_all(saveall)
print(newdf)
os.system("rm -rf /tmp/ak_save*")
ak.disconnect()
