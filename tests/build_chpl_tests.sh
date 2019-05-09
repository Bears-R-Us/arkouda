#!/bin/bash -v

export MY_CHPL_FLAGS="--print-passes --cache-remote -senableParScan --fast -M ../"


chpl $MY_CHPL_FLAGS UnitTestArgSortMsg.chpl
