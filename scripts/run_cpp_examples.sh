#!/usr/bin/env bash
# Runs all C++ examples

set -Eeuo pipefail

if [[ "$#" -gt 1 ]]; then
    echo "usage: run_cpp_examples.sh <prefix, e.g., mpirun -n 4 -oversubscribe>"
	exit 1
fi

PREFIX="${1:-} `pwd`/build/bin"
out=results/`git rev-parse HEAD`/cpp

for ex in bench brunel gap_junctions generators lfp ring single-cell "probe-demo v"
do
    echo "Running: $ex"
    dir=`echo $ex | tr ' ' '_'`
    mkdir -p $out/$dir
    cd $out/$dir
    $PREFIX/$ex > stdout.txt 2> stderr.txt
    cd -
done
