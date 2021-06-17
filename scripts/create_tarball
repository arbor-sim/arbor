#!/usr/bin/env bash
# creates a tar ball of Arbor for e.g. releases
#
# checks out repo at a specific branch/commit/tag
# strips version control,
# but leaves (empty) .git directories of submodules (CMake checks for it)

set -Eeuo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "usage: create_tarball.sh path_to_repo branch/commit/tag path_to_output_tarball"
	exit 1
fi

trap cleanup SIGINT SIGTERM ERR EXIT

cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  rm -rf "$TMP_DIR"
}

TMP_DIR=$(mktemp -d)

REPO=$1
CHECKOUT=$2
OUT_TARBALL=$3

if [[ ! "$TMP_DIR" || ! -d "$TMP_DIR" ]]; then
  echo "Could not create temporary directory"
  exit 1
fi

REPO_NAME=arbor

cd "$TMP_DIR"
git clone "$REPO" $REPO_NAME
cd "$REPO_NAME"
git checkout "$CHECKOUT"
git submodule init
git submodule update

# remove main .git
rm -vrf .git

# wipe .git submodule files
find -type f -name .git | xargs truncate -s 0

# create tar ball
cd ..
tar vcfz "$OUT_TARBALL" $REPO_NAME
