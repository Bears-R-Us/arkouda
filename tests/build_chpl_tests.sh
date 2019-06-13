#!/bin/bash -v

export MY_CHPL_FLAGS="--print-passes --cache-remote -senableParScan -suseBulkTransfer=true --fast -M ../"


chpl $MY_CHPL_FLAGS UnitTestArgSortMsg.chpl
chpl $MY_CHPL_FLAGS UnitTestargsortDRS.chpl
chpl $MY_CHPL_FLAGS UnitTestIn1d.chpl
