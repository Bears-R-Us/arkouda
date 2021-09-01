Arkouda scalability graphs for Chapel 1.23.0 vs 1.24.1 

Benchmark runs on up to 240 nodes of a Cray XC with Aries and HPE Apollo with
HDR-100 InfiniBand.

For HPE Apollo we're comparing Chapel 1.23.0 with Arkouda from 10/20/20
(https://github.com/Bears-R-Us/arkouda/commit/224a7c3) and Chapel 1.24.1 with
Arkouda from 04/06/21 (https://github.com/Bears-R-Us/arkouda/commit/795e7e2).

For Cray XC we're just running with Chapel 1.24.1 and Arkouda from 04/06/21 and
comparing that to the same configuration on Apollo.

There is data for stream, argsort, gather, and scatter using 8 GiB per node and
some larger argsort runs up to 256 GiB per node on Apollo.

Hardware Configuration:
---
Cray XC:
 - Aries network
 - 36-core Broadwell (dual-socket Intel E5-2695V4)
 - 128 GB Ram 

HPE Apollo:
 - 100 Gb HDR-100 InfiniBand network
 - 128-core Rome (dual-socket AMD EPYC 7702)
 - 2048 GB RAM

Software Configuration:
---
Cray XC:
 - CLE 7.0.UP02
 - cray-python 3.7.3.2
 - Chapel 1.24.1
 - GCC 10.2.0
 - ZeroMQ 4.3.2
 - HDF5 1.10.5

HPE Apollo:
 - RHEL 7.8
 - Python 3.8.2
 - Chapel 1.23.0 and 1.24.1
 - GCC 10.2.0
 - ZeroMQ 4.3.2
 - HDF5 1.10.5
