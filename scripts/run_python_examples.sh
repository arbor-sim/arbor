#!/usr/bin/env bash
# Runs all Python examples

set -Eeuox pipefail

if [[ "$#" -gt 1 ]]; then
    echo "usage: run_python_examples.sh <prefix>"
    exit 1
fi

PREFIX=${1:-}

$PREFIX python3 -m pip install -r python/example/example_requirements.txt -U

runpyex () {
  echo "=== Executing $1 ======================================"
  $PREFIX python3 python/example/$*
  echo "### Example $1: OK."
}

runpyex brunel.py -n 400 -m 100 -e 20 -p 0.1 -w 1.2 -d 1 -g 0.5 -l 5 -t 100 -s 0.5 -G 50 -S 123
# runpyex dynamic-catalogue.py # depends on running a-b-c first, see run_external_tests.sh
runpyex gap_junctions.py
runpyex single_cell_cable.py
runpyex single_cell_detailed_recipe.py python/example/single_cell_detailed.swc
runpyex single_cell_detailed.py python/example/single_cell_detailed.swc
runpyex probe_lfpykit.py python/example/single_cell_detailed.swc
runpyex single_cell_model.py
runpyex single_cell_nml.py python/example/morph.nml
runpyex single_cell_recipe.py
runpyex single_cell_stdp.py
runpyex single_cell_swc.py python/example/single_cell_detailed.swc
runpyex network_ring.py
# runpyex network_ring_mpi.py # requires MPI
# runpyex network_ring_mpi_plot.py # no need to test
runpyex network_ring_gpu.py # by default, gpu_id=None
runpyex network_two_cells_gap_junctions.py
runpyex network_ring.py
runpyex diffusion.py
runpyex plasticity.py
runpyex v-clamp.py
runpyex calcium_stdp.py
