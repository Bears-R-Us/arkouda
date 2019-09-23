#!/bin/bash
# Generate `chpl` compiler flags for broader version compatibility.
# This script optionally accepts the path to a `chpl` compiler.
set -o errexit -o pipefail -o noclobber -o nounset

CHPL=chpl
if (( $# >= 1 )); then
    CHPL="$1"
fi

re='(([[:digit:]]+)\.([[:digit:]]+)\.([[:digit:]]+))'
[[ "$($CHPL --version | head -n 1)" =~ $re ]]
CHPL_VERSION="${BASH_REMATCH[1]}"
CHPL_MAJOR_VERSION="${BASH_REMATCH[2]}"
CHPL_MINOR_VERSION="${BASH_REMATCH[3]}"
CHPL_PATCH_VERSION="${BASH_REMATCH[4]}"

FLAGS=

if (( CHPL_MAJOR_VERSION >= 1 )) && (( CHPL_MINOR_VERSION >= 20 )); then
    FLAGS+=" --legacy-classes "
fi

echo $FLAGS
