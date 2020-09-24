Arkouda scalability runs

Benchmarks run on up to 512 nodes of a Cray-XC with Aries and 32 nodes of a
Cray-CS With FDR InfiniBand. Per node hardware is similar. Using Arkouda
from 04/16/20: https://github.com/mhmerrill/arkouda/commit/f8a7422. Chapel
1.20.0 and 1.21.0 were used, and a second 1.20.0 run with aggregation disabled
was also run.

There is data for stream.py, argsort.py, gather.py, and scatter.py using the
default problem size of ~3/4 GB per node.

Results 

Cray-XC Configuration:
---
 - 36-core (72 HT), 128 GB RAM
   - dual 18-core (36 HT) "Broadwell" 2.1 GHz processors
 - Aries network
 - Software:
  - CLE 7.0.UP02
  - cray-python 3.7.3.2
  - Chapel 1.20.0, 1.21.0
  - ZeroMQ 4.3.2
  - cray-hdf5 1.10.5.2


Cray-CS Configuration:
---
 - 36-core (72 HT), 128 GB RAM
   - dual 18-core (36 HT) "Broadwell" 2.1 GHz processors
 - 56 Gb FDR InfiniBand network
 - Software:
   - RHEL 7.6
   - Python 3.6.8
   - Chapel 1.20.0, 1.21.0
   - ZeroMQ 4.3.2
   - HDF5 1.10.5


Dir structure:
 - .              -- contains combined .gpi and generated .pdfs
 - cray-{cs,xc}
   - graphs/      -- contains .gpi files and generated .pdfs
     - data/      -- contains data collated into .dat files
