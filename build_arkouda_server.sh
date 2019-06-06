#!/bin/bash

export MY_CHPL_FLAGS="--ccflags=-Wno-incompatible-pointer-types --print-passes --cache-remote -senableParScan -suseBulkTransfer=true --instantiate-max 512 --fast"

echo "chpl $MY_CHPL_FLAGS arkouda_server.chpl"

chpl $MY_CHPL_FLAGS arkouda_server.chpl
