#!/usr/bin/env python3
# coding: utf-8

# In[13]:


import importlib
import sys, time
from glob import glob
import arkouda as ak


# In[14]:


importlib.reload(ak)


# In[15]:


ak.set_defaults()
ak.v = False
if len(sys.argv) > 1:
    ak.connect(server=sys.argv[1], port=sys.argv[2])
    hdffiles = [sys.argv[3]]
else:
    print("usage: lanl_io_test server port file")
    sys.exit(1)

fields = ['srcIP', 'dstIP', 'srcPort', 'dstPort', 'start']

nfDF = {field: ak.read_hdf(field, hdffiles) for field in fields}

print(nfDF['start'])
print(nfDF['srcIP'])
print(nfDF['dstIP'])
print(nfDF['srcPort'])
print(nfDF['dstPort'])
print(nfDF)

print(ak.info(ak.AllSymbols))

u,c = nfDF['srcIP'].unique(return_counts=True)
print(u.size,u)
print(c.size,c)

u,c = nfDF['dstIP'].unique(return_counts=True)
print(u.size,u)
print(c.size,c)

u,c = nfDF['srcPort'].unique(return_counts=True)
print(u.size,u)
print(c.size,c)

u,c = nfDF['dstPort'].unique(return_counts=True)
print(u.size,u)
print(c.size,c)

ak.shutdown()



