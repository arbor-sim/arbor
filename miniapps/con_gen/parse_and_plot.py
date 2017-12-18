#!/usr/bin/env python
"""
Simple plotting script for plotting a synapse file as produced by 
./miniapps/con_gen/con_gen.exe
"""
import matplotlib.pyplot as plt
import collections
import math
import sys

default_synapse_path = "./synapses.dat"
multiplier = 30.0

def usage():
    print """
Simple plotting script for plotting a synapse file as produced by 
./miniapps/con_gen/con_gen.exe

When called without parameters it will plot the connections as produced by con_gen.exe
with default settings. Only the first gids of the two first populations will
be parsed and showed


Usage:
python parse_and_plot.py [path_synapse path_populations]

Arguments:
 path_synapse       path to file with synapses to plot
 path_populations   path to population file used in con_gen.exe

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

def plot_gids_in_space(gids, weights, delays, populations):
    """
    Plot the histogram and target curve in one figure
    """
    side_x = populations[0][0]
    side_y = populations[0][1]

    population_1_size = populations[0][0] * populations[0][1]
    population_2_size = populations[1][0] * populations[1][1]
    max_gid = population_1_size + population_2_size

    gid_count = collections.defaultdict(int)
    for gid, weight in zip(gids, weights):
        # Skip gids from population higher then the 2nd
        if gid > max_gid:
            continue
        gid_count[gid] += 1

    fig, ax = plt.subplots(figsize=(side_x/10, side_y/10))

    for gid, count in gid_count.items():
        # convert to location in space
        if gid >= population_1_size:
            gid -= population_1_size
            marker = 'ro' 
            zorder = 0         
        else:
            marker = 'go'
            zorder = 1
            
        x = gid % side_x
        y = gid / side_x

        l = ax.plot(x, y, marker)
        plt.setp(l, 'markersize', count, zorder=zorder)

    plt.xlim((-0.5, side_x - 0.5))
    plt.ylim((-0.5, side_y - 0.5))

    plt.title("Pre synaptic neurons for selected gids between two (green / red) populations")
    plt.xlabel("Cell position grid_x")
    plt.ylabel("Cell position grid_y")
    plt.show()


def parse_population_path(path):
    populations = []
    with open(path , "r") as f:
        for line in f:
            tokens = line.split(",")
            try:
                size_x = int(tokens[0].strip())
                size_y = int(tokens[1].strip())
                periodic = evaluate_string_true(tokens[2].strip())
            except:
                print "Failed parsing:", tokens[0], ", ", tokens[1]               
                exit(2)

            populations.append([size_x, size_y, periodic])

    return populations


def evaluate_string_true(s="True"):
    """
    Evaluates if s is a string with True value
    True -> 'true', '1', 't', 'yes', 'y'
    Capitalization is ignored
    """
    return s.lower() in ['true', '1', 't', 'yes', 'y']

def main(synapse_path=default_synapse_path, population_path=None):
    """

    """

    [gids, weights, delays] = parse_gid_file(synapse_path)

    # plot the binned spikes and the optional curve
    if population_path==None:
        populations = [[100,100,True],[100, 100, True]]
        plot_gids_in_space(gids, weights, delays, populations)
    else:
        populations = parse_population_path(population_path)
        plot_gids_in_space(gids, weights, delays, populations)



if __name__ == "__main__":
    # Non checked command line parsing
    spike_file_path = ""
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2] )
    else:
        usage()
        exit(1)
