#!/bin/bash

# Chapel clone and build script

# TODO: check for brew install if needed
# TODO: check for git, install with brew if needed

# can't use brew install chapel b/c doesn't support remake
git clone https://github.com/chapel-lang/chapel.git --branch release/1.20
cd chapel
source util/setchplenv.bash

# setup exports
export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=smp
export CHPL_GASNET_CFG_OPTIONS=--disable-ibv
export CHPL_TARGET_CPU=native
export GASNET_SPAWNFN=L
export GASNET_ROUTE_OUTPUT=0
export GASNET_QUIET=Y
export GASNET_MASTERIP=127.0.0.1

# Set these to help with oversubscription
export QT_AFFINITY=no
export CHPL_QTHREAD_ENABLE_OVERSUBSCRIPTION=1

make
make check
