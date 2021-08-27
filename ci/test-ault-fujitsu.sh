#! /bin/bash

base_path="$1"
host="$(hostname)"
echo "=== path: $base_path"
echo "=== host: $host"

echo "=== loading environment"
source /users/bcumming/a64fx/env.sh
spack load git gcc@11.1.0 cmake ninja

build_path="$base_path/build"
echo "=== building: $build_path"
mkdir "$build_path"
cd "$build_path"
echo "=== CC=gcc CXX=g++ cmake .. -DARB_USE_BUNDLED_LIBS=on -DARB_ARCH=armv8.2-a+sve -DARB_VECTORIZE=on -G Ninja"
CC=gcc CXX=g++ cmake .. -DARB_USE_BUNDLED_LIBS=on -DARB_ARCH=armv8.2-a+sve -DARB_VECTORIZE=on -G Ninja
ninja -j 8 unit

bin_path="$build_path/bin"
echo "=== running unit tests: $bin_path/unit"
cd "$bin_path"
./unit

