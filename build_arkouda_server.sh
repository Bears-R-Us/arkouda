#!/bin/bash

export MY_CHPL_FLAGS="--ccflags=-Wno-incompatible-pointer-types --print-passes --cache-remote --instantiate-max 1024 --fast"

echo "chpl $MY_CHPL_FLAGS arkouda_server.chpl"

chpl $MY_CHPL_FLAGS arkouda_server.chpl
