#!/usr/bin/env bash
# Runs all C++ examples

set -Eeuo pipefail

if [[ "$#" -gt 1 ]]; then
    echo "usage: run_cpp_examples.sh <prefix, e.g., mpirun -n 4 -oversubscribe>"
	exit 1
fi

PREFIX=${1:-}

$PREFIX build/bin/bench
$PREFIX build/bin/brunel
$PREFIX build/bin/dryrun
$PREFIX build/bin/gap_junctions
$PREFIX build/bin/generators
$PREFIX build/bin/lfp
$PREFIX build/bin/probe-demo v
$PREFIX build/bin/ring
$PREFIX build/bin/single-cell
