#!/bin/bash -v

export MY_CHPL_FLAGS="--print-passes --cache-remote -senableParScan --fast"


chpl $MY_CHPL_FLAGS arkouda_server.chpl
