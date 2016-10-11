#!/usr/bin/env bash

# NB: this is a convenience script; under normal circumstances
# validation files will be built as required by CMake.

cmd="${0##*/}"
function usage {
    echo "usage: $cmd SCRIPTDIR [DESTDIR]"
    exit 1
}

if [ $# -gt 2 -o $# -lt 1 ]; then usage; fi

scriptdir="$1"
if [ ! -d "$scriptdir" ]; then
    echo "$cmd: no such directory '$scriptdir'"
    usage
fi

destdir="."
if [ $# -eq 2 ]; then destdir="$2"; fi
if [ ! -d "$destdir" ]; then
    echo "$cmd: no such directory '$destdir'"
    usage
fi

for v in soma ball_and_stick ball_and_squiggle ball_and_taper ball_and_3stick simple_exp_synapse simple_exp2_synapse; do
    (cd "$scriptdir"; nrniv -nobanner -python ./$v.py) > "$destdir/neuron_$v.json" &
done
wait

