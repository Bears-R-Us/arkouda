Arkouda scalability runs for DstAggregator optimization

Benchmarks run on up to 256 nodes of a Cray-XC with Aries, 32 nodes of a
Cray-CS with FDR InfiniBand, and 32 nodes of a Cray-CS with HDR InfiniBand.
Comparing perf before and after https://github.com/mhmerrill/arkouda/pull/485
with Chapel 1.22.

There is data for scatter.py and argsort.py using the default problem size of
~3/4 GiB per node and a larger problem size with 8 GiB per node.

Cray-XC Configuration:
---
 - 36-core (72 HT), 128 GB RAM
   - dual 18-core (36 HT) "Broadwell" 2.1 GHz processors
 - Aries network

Cray-CS Configuration:
---
 - 36-core (72 HT), 128 GB RAM
   - dual 18-core (36 HT) "Broadwell" 2.1 GHz processors
 - 56 Gb FDR InfiniBand network

Cray-CS Configuration:
---
 - 128-core (256 HT), 256 GB RAM
   - dual 64-core (128 HT) "Rome" 2.0 GHz processors
 - 200 Gb FDR InfiniBand network
