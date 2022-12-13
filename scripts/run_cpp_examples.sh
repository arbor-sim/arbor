#!/usr/bin/env bash
# Runs all C++ examples

set -Eeuo pipefail

if [[ "$#" -gt 1 ]]; then
    echo "usage: run_cpp_examples.sh <prefix, e.g., mpirun -n 4 -oversubscribe>"
	exit 1
fi

PREFIX="${1:-} `pwd`/build/bin"
tag=dev-`git rev-parse --short HEAD`
out="results/$tag/cpp/"

ok=0
check () {
    prog=$1
    expected="$2 spikes"
    actual=$(grep -Eo '[0-9]+ spikes' $out/$prog/stdout.txt || echo "N/A")
    if [ "$expected" == "$actual" ]
    then
        echo "   - $prog: OK"
    else
        echo "   - $prog: ERROR wrong number of spikes: $expected ./. $actual"
        ok=1
    fi
}

for ex in bench brunel gap_junctions generators lfp ring single-cell "probe-demo v" plasticity ou voltage-clamp
do
    echo "   - $ex"
    dir=`echo $ex | tr ' ' '_'`
    mkdir -p $out/$dir
    cd $out/$dir
    $PREFIX/$ex > stdout.txt 2> stderr.txt
    cd -
done

# Do some sanity checks.
check brunel 6998
check bench 972
check ring 94
check gap_junctions 30

exit $ok
