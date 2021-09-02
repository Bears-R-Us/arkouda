Arkouda scalability graphs for InfiniBand CQ serialization fix

This includes benchmark results on up to 240 nodes of an HPE Apollo system with
HDR-100 InfiniBand. This compares performance between the official Chapel
1.24.1 release and that release with a patch to serialize InfiniBand completion
queue polling, which reduces contention. That patch was eventually merged into
Chapel in https://github.com/chapel-lang/chapel/pull/18243. Runs are using
Arkouda from 04/06/21 (https://github.com/Bears-R-Us/arkouda/commit/795e7e2)

There is data for stream, gather, scatter, and argsort using 8 GiB per node and
some larger sort runs up to 256 GiB.

Hardware Configuration:
---
HPE Apollo:
 - 100 Gb HDR-100 InfiniBand network
 - 128-core Rome (dual-socket AMD EPYC 7702)
 - 2048 GB RAM

Software Configuration:
---
HPE Apollo:
 - RHEL 7.8
 - Python 3.8.2
 - Chapel 1.24.1 (w/o and w/ serialization fix)
 - GCC 10.2.0
 - ZeroMQ 4.3.2
 - HDF5 1.10.5
