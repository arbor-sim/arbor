#!/usr/bin/env bash
# Runs all C++ examples

set -Eeuo pipefail

if [[ "$#" -gt 1 ]]; then
    echo "usage: run_cpp_examples.sh <prefix, e.g., mpirun -n 4 -oversubscribe>"
	exit 1
fi

PREFIX="${1:-} `pwd`/build/bin"

cxx=/usr/local/opt/llvm/bin/clang++
cc=/usr/local/opt/llvm/bin/clang

for tag in v0.4 v0.5.2 v0.6 v0.7 v0.8
do
    echo "Version=$tag"
    rm -rf ext/*
    git checkout $tag
    git checkout $tag -- ext/
    git submodule update --init
    for simd in ON OFF
    do
        echo " * simd=$simd"
        out=results/$tag-`git rev-parse --short HEAD`/cpp/simd=$simd
        cd build
        cmake .. -DARB_USE_BUNDLED_LIBS=ON -DCMAKE_CXX_COMPILER=$cxx -DCMAKE_C_COMPILER=$cc -DCMAKE_BUILD_TYPE=release -DARB_VECTORIZE=$simd -DARB_ARCH=native
        ninja install examples
        cd -
        for ex in bench brunel gap_junctions generators lfp ring single-cell "probe-demo v"
        do
            echo "   - $ex"
            dir=`echo $ex | tr ' ' '_'`
            mkdir -p $out/$dir
            cd $out/$dir
            $PREFIX/$ex > stdout.txt 2> stderr.txt
            cd -
        done
    done
done

ok=0
check () {
    prog=$1
    expected="$2 spikes"
    actual=$(/usr/bin/grep -Eo '\d+ spikes' results/$tag/cpp/SIMD=$simd/$prog/stdout.txt || echo "N/A")
    if [ "$expected" == "$actual" ]
    then
        echo "   - $prog: OK"
    else
        echo "   - $prog: ERROR wrong number of spikes: $expected ./. $actual"
        ok=1
    fi
}

for tag in "v0.4-79855b66" "v0.5.2-51e35898" "v0.6-930c23eb" "v0.7-d0e424b4" "v0.8-8e82ec1"
do
    echo "Version=$tag"
    for simd in ON OFF
    do
        echo " * SIMD=$simd"
        check brunel 6998
        check bench 972
        if [[ "$tag" < "v0.7" ]]
        then
            check ring 19
        else
            check ring 94
        fi
        check gap_junctions 30
    done
done
exit $ok
