#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import json
import math
import sys

# Read JSON output of lfp example from stdin and plot.
#
# The JSON data for timeseries is structured as:
#    <string>: { "unit": <string>, "time": [ <number>... ], "value": [ <number>... ] }
# or:
#    <string>: { "unit": <string>, "time": [ <number>... ], "values": [[ <number>... ] ...] }
#
# 2-d morphology data is represented as samples (x, z, r) where r is the radius, one array
# per branch. Extra point data (probe location, electrode sites) as pairs (x, z):
#   "morphology": { "unit": <string>, "samples":  [[[<number> <number> <number>] ...] ...]
#                   "probe": [<number> <number>], "electrodes": [[<number> <number>] ...] }


def subplot_timeseries(fig, index, jdict, key):
    data = jdict[key]
    sub = fig.add_subplot(index, ylabel=data["unit"], title=key, xlabel="time (ms)")
    ts = data["time"]
    vss = data["values"] if "values" in data else [data["value"]]

    for vs in vss:
        sub.plot(ts, vs)


def subplot_morphology(fig, index, jdict, key, xlim, ylim):
    data = jdict[key]
    unit = data["unit"]
    sub = fig.add_subplot(
        index,
        xlabel="x (" + unit + ")",
        ylabel="y (" + unit + ")",
        title=key,
        xlim=xlim,
        ylim=ylim,
    )

    for samples in data["samples"]:
        polys = [
            (
                [x0 - s0 * dy, x0 + s0 * dy, x1 + s1 * dy, x1 - s1 * dy],
                [y0 + s0 * dx, y0 - s0 * dx, y1 - s1 * dx, y1 + s1 * dx],
            )
            for ((x0, y0, r0), (x1, y1, r1)) in zip(samples, samples[1:])
            for dx, dy in [(x1 - x0, y1 - y0)]
            for d in [math.sqrt(dx * dx + dy * dy)]
            if d > 0
            for s0, s1 in [(r0 / d, r1 / d)]
        ]

        for xs, ys in polys:
            sub.fill(xs, ys, "k")
    sub.plot(*[u for x, y in data["electrodes"] for u in [[x], [y], "o"]])
    sub.plot(*[u for x, y in [data["probe"]] for u in [[x], [y], "r*"]])


P = argparse.ArgumentParser(description="Plot results of LFP demo.")
P.add_argument(
    "input",
    metavar="FILE",
    nargs="?",
    type=argparse.FileType("r"),
    default=sys.stdin,
    help="LFP example output in JSON",
)
P.add_argument(
    "-o", "--output", metavar="FILE", dest="outfile", help="save plot to file FILE"
)

args = P.parse_args()
j = json.load(args.input)

fig = plt.figure(figsize=(9, 5))
fig.subplots_adjust(wspace=0.6, hspace=0.9)

subplot_morphology(fig, 131, j, "morphology", xlim=[-100, 100], ylim=[-100, 600])
subplot_timeseries(fig, 332, j, "synaptic current")
subplot_timeseries(fig, 335, j, "membrane potential")
subplot_timeseries(fig, 338, j, "ionic current density")
subplot_timeseries(fig, 133, j, "extracellular potential")

if args.outfile:
    fig.savefig(args.outfile)
else:
    plt.show()

plt.close(fig)
