#!/bin/bash
# Generate `chpl` compiler flags for broader version compatibility.
set -o errexit -o pipefail -o noclobber -o nounset

re='(([[:digit:]]+)\.([[:digit:]]+)\.([[:digit:]]+))'
[[ "$(chpl --version | head -n 1)" =~ $re ]]
CHPL_VERSION="${BASH_REMATCH[1]}"
CHPL_MAJOR_VERSION="${BASH_REMATCH[2]}"
CHPL_MINOR_VERSION="${BASH_REMATCH[3]}"
CHPL_PATCH_VERSION="${BASH_REMATCH[4]}"

FLAGS=

if (( CHPL_MAJOR_VERSION >= 1 )) && (( CHPL_MINOR_VERSION >= 20 )); then
    FLAGS+=" --legacy-classes "
fi

echo $FLAGS
