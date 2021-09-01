Arkouda argsort hero runs on HPE Apollo

This includes some "hero" argsort results on 576 nodes of an HPE Apollo system
with HDR-100 InfiniBand. Results are with Chapel 1.24.1 with an InfiniBand
completion queue serialization fix applied and Arkouda from 04/06/21
(https://github.com/Bears-R-Us/arkouda/commit/795e7e2)

Hardware Configuration:
---
HPE Apollo:
 - 100 Gb HDR-100 InfiniBand network
 - 128-core Rome (dual-socket AMD EPYC 7702)
 - 1024 GB RAM

Software Configuration:
---
HPE Apollo:
 - RHEL 7.8
 - Python 3.8.2
 - Chapel 1.24.1 (w/ serialization fix)
 - GCC 10.2.0
 - ZeroMQ 4.3.2
 - HDF5 1.10.5
