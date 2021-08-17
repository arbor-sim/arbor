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

ARBOR_DIR=$TMP_DIR/arbor
mkdir $ARBOR_DIR
cp -r $ARBOR_SOURCE/* $ARBOR_DIR

cd "$TMP_DIR"

SPACK_DIR=spack
SPACK_REPO=https://github.com/spack/spack
SPACK_CUSTOM_REPO=custom_repo

SPACK_VERSION=$2 # latest_release or develop
SPACK_BRANCH=develop # only used for develop

case $SPACK_VERSION in
    "develop")
        git clone --depth 1 --branch $SPACK_BRANCH $SPACK_REPO $SPACK_DIR
        ;;
    "latest_release")
        wget "$(curl -sH "Accept: application/vnd.github.v3+json" https://api.github.com/repos/spack/spack/releases/latest | grep browser_download_url |  cut -d '"' -f 4)"
        tar xfz spack*.tar.gz
        ln -s spack*/ $SPACK_DIR
        ;;
    *)
        echo "SPACK_VERSION" must be \"latest_release\" or \"develop\"
        exit 1
esac

mkdir ~/.spack
cp $ARBOR_DIR/spack/config.yaml ~/.spack

source $SPACK_DIR/share/spack/setup-env.sh
spack repo create $SPACK_CUSTOM_REPO

mkdir -p $SPACK_CUSTOM_REPO/packages/arbor
spack repo add $SPACK_CUSTOM_REPO

# to make use of the cached installations
spack reindex

cp $ARBOR_DIR/spack/package.py $SPACK_CUSTOM_REPO/packages/arbor
cd $ARBOR_DIR
ARBOR_VERSION=$(cat "$ARBOR_DIR/VERSION")
spack dev-build arbor@${ARBOR_VERSION}
