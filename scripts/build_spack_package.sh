#!/usr/bin/env bash
# checks out Spack and Arbor and builds it with the package.py from Arbor's repo
# Spack can be the latest release or the develop branch

set -Eeuo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "Builds the in-repo Spack package of Arbor against the latest Spack release or a given Spack branch"
    echo "usage: build_spack_package.sh arbor_source_directory latest_release|develop"
    exit 1
fi

trap cleanup SIGINT SIGTERM ERR EXIT

cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  rm -rf "$TMP_DIR"
}

TMP_DIR=$(mktemp -d)
ARBOR_SOURCE=$1

cd "$TMP_DIR"
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
# ./spack/bin/spack install libelf
. spack/share/spack/setup-env.sh
cp ./spack/package.py spack/var/spack/repos/builtin/packages/arbor/

spack env create -d ./spack_env
spacktivate ./spack_env
spack add arbor
ARBOR_VERSION=$(cat "$ARBOR_SOURCE/VERSION")
spack develop --path "$ARBOR_SOURCE" --no-clone arbor@${ARBOR_VERSION} +python
spack concretize -f
spack install
