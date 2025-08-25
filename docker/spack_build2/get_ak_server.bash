#!/bin/bash
# get_ak_server.bash

# source spack setup-env.sh
source /opt/spack-develop/share/spack/setup-env.sh

# Use spack location to get the package installation path
ak_server=$(spack location -i arkouda)/bin/arkouda_server

echo ${ak_server}
