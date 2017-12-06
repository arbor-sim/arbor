#!/usr/bin/env python
"""
Simple (non automatic) validation script of the inhomgeneous Poisson Spike Source
cell.
"""
import matplotlib.pyplot as plt
import collections
import math
import sys

default_synapse_path = "./synapses.dat"
multiplier = 30.0

def usage():
    print """
Usage:
python parse_and_plot.py

Arguments:
 -path_spikes    blabla

Bla bla bla
"""

def parse_gid_file(path):
    """
    Simple spike file parsing function. Assumes no errors and might fail
    silently on errors
    """
    gids = []
    weights = []
    delays = []
    with open(path , "r") as f:
        for line in f:
            tokens = line.split(",")
            try:
                gid = int(tokens[0].strip())
                weight = float(tokens[1].strip())
                delay = float(tokens[2].strip())
            except:
                print "Failed parsing:", tokens[0]
                exit(2)

            gids.append(gid)
            weights.append(weight)
            delays.append(delay)

    return gids, weights, delays

def plot_gids_in_space(gids, weights, delays):
    """
    Plot the histogram and target curve in one figure
    """
    side = 100


    gid_count_h = collections.defaultdict(int)
    gid_count_l = collections.defaultdict(int)


    for gid, weight in zip(gids, weights):

        if weight > 10.0:
            gid_count_h[gid] += 1
        else:
            gid_count_l[gid] += 1

    for gid, count in gid_count_l.items():
        # convert to location in space
        if gid >= 10000:
            gid -= 10000
        x = gid % side
        y = gid / side

        l = plt.plot(x, y, 'ro')
        plt.setp(l, 'markersize', count)

    for gid, count in gid_count_h.items():
        # convert to location in space
        if gid >= 10000:
            gid -= 10000
        x = gid % side
        y = gid / side

        l = plt.plot(x, y, 'go')
        plt.setp(l, 'markersize', count)

    plt.xlim((0,100))
    plt.ylim((0,100))

    #plt.title("Instantanious spike rate for a population of ipss cells ")
    #plt.xlabel("Time (ms.)")
    #plt.ylabel("Instantanious spike rate (spikes / second)")
    plt.show()

def main():
    """

    """

    [gids, weights, delays] = parse_gid_file(default_synapse_path)




    # plot the binned spikes and the optional curve
    plot_gids_in_space(gids, weights, delays)

def evaluate_string_true(s="True"):
    """
    Evaluates if s is a string with True value
    True -> 'true', '1', 't', 'yes', 'y'
    Capitalization is ignored
    """
    return s.lower() in ['true', '1', 't', 'yes', 'y']

if __name__ == "__main__":
    # Non checked command line parsing
    spike_file_path = ""
    if len(sys.argv) == 1:
        main()
    else:
        usage()
        exit(1)
