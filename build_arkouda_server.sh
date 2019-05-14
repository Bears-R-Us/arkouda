#!/bin/bash -v

export MY_CHPL_FLAGS="--ccflags=-Wno-incompatible-pointer-types --print-passes --cache-remote -senableParScan --fast"


chpl $MY_CHPL_FLAGS arkouda_server.chpl
