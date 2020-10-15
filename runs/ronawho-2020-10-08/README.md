Arkouda scalability runs for Chapel 1.22 vs 1.23

This is intended to see performance changes for both Chapel and Arkouda over
the last 6 months as well as compare current performance across systems using
current Chapel/Arkouda.

Benchmarks run on:
 - 256 node Cray XC (Aries) with 36-cores per node
 - 32 node Cray CS (FDR IB) with 36-cores per node
 - 32 node Cray CS (HDR IB) with 128-cores per node

There are runs for:
 - Chapel 1.22 with 04/16/20 Arkouda (https://github.com/mhmerrill/arkouda/commit/f8a7422)
 - Chapel 1.22 with 10/06/20 Arkouda (https://github.com/mhmerrill/arkouda/commit/bc06135)
 - Chapel 1.23 with 10/06/20 Arkouda (https://github.com/mhmerrill/arkouda/commit/bc06135)

There is data for gather, scatter, argsort, stream, scan, and reduce using the
default problem size (~3/4 GiB) and a large problem size (8 GiB).


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
 - 200 Gb HDR InfiniBand network
