Arkouda scalability runs

Benchmarks run on up to 512 nodes of a Cray-XC with Aries and 32 nodes of a
Cray-CS With FDR InfiniBand. Per node hardware is similar. Using Arkouda
from 12/12/19: https://github.com/mhmerrill/arkouda/commit/da0a124

Ran stream.py, argsort.py, gather.py, scatter.py, scan.py, and reduce.py. The
data for scan/reduce isn't collated or graphed, but the raw data is available.
The default problem size ~3/4 GB per node was run for both CS and XC. For XC
there are also runs that use 16 GB per node (8TB at 512 nodes.)

Cray-XC Configuration:
---
 - 36-core (72 HT), 128 GB RAM
   - dual 18-core (36 HT) "Broadwell" 2.1 GHz processors
 - Aries network
 - Software:
  - CLE 7.0.UP02
  - cray-python 3.7.3.2
  - Chapel 1.20.0
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
   - Chapel 1.20.0
   - ZeroMQ 4.3.2
   - HDF5 1.10.5


Dir structure:
 - .              -- contains combined .gpi and generated .pdfs
 - cray-{cs,xc}
   - graphs/      -- contains .gpi files and generated .pdfs
     - data/      -- contains data collated into .dat files
     - raw-data/  -- contains output from runs
       - print.sh -- grep time/perf (modify for size and time/perf)
       - run.sh   -- script I used to run problem sizes
