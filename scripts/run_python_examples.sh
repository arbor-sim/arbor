#!/usr/bin/env bash
# Runs all Python examples

set -Eeuo pipefail

if [[ "$#" -gt 1 ]]; then
    echo "usage: run_python_examples.sh <prefix>"
    exit 1
fi

PREFIX=${1:-}

$PREFIX python -m pip install -r python/example/example_requirements.txt

$PREFIX python python/example/brunel.py -n 400 -m 100 -e 20 -p 0.1 -w 1.2 -d 1 -g 0.5 -l 5 -t 100 -s 1 -G 50 -S 123
# $PREFIX python python/dynamic-catalogue.py # arbor-build-catalog is a test already
$PREFIX python python/example/gap_junctions.py
$PREFIX python python/example/single_cell_cable.py
$PREFIX python python/example/single_cell_detailed_recipe.py python/example/single_cell_detailed.swc
$PREFIX python python/example/single_cell_detailed.py python/example/single_cell_detailed.swc
$PREFIX python python/example/probe_lfpykit.py python/example/single_cell_detailed.swc
$PREFIX python python/example/single_cell_model.py
$PREFIX python python/example/single_cell_nml.py python/example/morph.nml
$PREFIX python python/example/single_cell_recipe.py
$PREFIX python python/example/single_cell_stdp.py
$PREFIX python python/example/single_cell_swc.py python/example/single_cell_detailed.swc
$PREFIX python python/example/network_ring.py
# $PREFIX python python/example/network_ring_mpi.py # requires MPI
# $PREFIX python python/example/network_ring_mpi_plot.py # no need to test
$PREFIX python python/example/network_ring_gpu.py # by default, gpu_id=None
$PREFIX python python/example/network_two_cells_gap_junctions.py
$PREFIX python python/example/diffusion.py
