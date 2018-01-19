#!/usr/bin/env python
"""
Simple (non automatic) validation script of the inhomgeneous Poisson Spike Source
cell.
"""
import matplotlib.pyplot as plt
import collections
import math
import sys

default_spike_file_path = "./spikes.gdf"
multiplier = 30.0

def usage():
    print """
Usage: 
python parse_and_plot.py [path_spikes] [path_time_rate] [plot_target interpolate]

Arguments:
 -path_spikes    (../spikes)path to gdf formatted file with spikes
 -path_time_rate file path with float, float pairs per line with time rate pairs
 -plot_target    Plot the (expected) target rates as a line
 -interpolate    Interpolate between the time-rate pairs

parse_and_plot.py is a simple script to validate the performance of the 
ipss cell. No input validation is performed.

- Parse gdf spike file
- Performs a binning operation for the first 1000 ms 
  - with 1 ms binning 
  - averages the found rates assuming a total of 10000 cells.
- Displays a figure with the histogram of the instantatious spike rate for
the whole group, including a target rate.

Default target rate: 
 hz.|                     
 240|     _-_             
    |    -   -  -         
    |   -     -- -        
 0  |__-__________-__     
      100        900   ms 

Todo:
 - Add statistical test for correct output for automated validation
"""

default_time_rate_pairs = [[0.0   , 0.0 * multiplier],
                           [50.0  , 0.0 * multiplier],
                           [100.0 , 1.0 * multiplier],
                           [200.0 , 5.0 * multiplier],
                           [300.0 , 7.0 * multiplier],
                           [400.0 , 8.0 * multiplier],
                           [500.0 , 7.0 * multiplier],
                           [600.0 , 3.0 * multiplier],
                           [700.0 , 3.0 * multiplier],
                           [750.0 , 5.0 * multiplier],
                           [800.0 , 2.5 * multiplier],
                           [900.0 , 0.0 * multiplier],
                           [1000.0, 0.0 * multiplier]]

def parse_spike_file(path):
    """ 
    Simple spike file parsing function. Assumes no errors and might fail 
    silently on errors
    """
    spikes_per_cell = collections.defaultdict(list)
    with open(path , "r") as f:
        for line in f:
            tokens = line.split(",")
            try:
                gid = int(tokens[0].strip())
                time = float(tokens[1].strip())
            except:
                print "Failed parsing:", tokens[0], ", ", tokens[1]               
                exit(2)

            spikes_per_cell[gid].append(time)

    return spikes_per_cell

def parse_time_rate(path):
    """ 
    Simple spike file parsing function. Assumes no errors and might fail 
    silently on errors
    """

    time_rate = []

    with open(path , "r") as f:
        for line in f:
            tokens = line.split(",")
            try:
                time = float(tokens[0].strip())
                rate = float(tokens[1].strip())
            except:
                print "Failed parsing:", tokens[0], ", ", tokens[1]               
                exit(2)

            time_rate.append([time, rate])

    return time_rate

def binning(spikes_per_cell, bin_size = 1.0, duration = 1000.0):
    """
    Perform a simple binning on the collected spikes before duration
    Assuming the times are in ms resolution
    bin_size (ms)
    duration (ms)
    """
    n_bins = int(math.floor(duration / bin_size))
    bins=[0]*n_bins

    for gid, spikes in spikes_per_cell.items():
        for spike in spikes:
            # Skip late spikes
            if spike > duration:
                continue

            # Convert to bin idx and increase 
            bins[int(math.floor(spike / bin_size))] += 1

    # normalize to spikes/second. 
    # we have 10000 cells but are binning per 1000
    # so devide by 10
    bins = [x / 10 for x in bins]

    return bins  

def plot_histogram_and_target_curve(bins, times=None, rates=None, plot_target=True):
    """
    Plot the histogram and target curve in one figure
    """
    plt.hist([idx for idx in range(len(bins))], len(bins), weights = bins)
    if plot_target:
        plt.plot(times, rates, linewidth=7,alpha=0.7)

    plt.title("Instantanious spike rate for a population of ipss cells ")
    plt.xlabel("Time (ms.)")
    plt.ylabel("Instantanious spike rate (spikes / second)")
    plt.show()

def main(path_spikes=None, path_time_rate=None, plot_target=True, interpolate=True):
    """
    Simple main function
    - Parse the spikes (expected in gdf format) from path 
      if path == None default path './spikes.gdf' will be used
    - Bin the spikes in 1000 ms bins
    - Plot a histogram
    - Plot a line with the default / expected rate if path is supplied
    todo:
    - Use the power of statistics to make this an automatic test
    """
    plot_default_curve = True

    # If no path supplied 
    if path_spikes == None:
        path_spikes = default_spike_file_path

    # If no path supplied 
    if not path_time_rate == None:
        time_rates = parse_time_rate(path_time_rate)
    else:
        time_rates = default_time_rate_pairs


    # parse the spikes
    spikes_per_cell = parse_spike_file(path_spikes)

    # perform binnen
    bins = binning(spikes_per_cell)
    
    # convert time_rate pairs to two lists
    # if we are NOT interpolating we need to fix the time_rates pairs
    if not interpolate:
        time_rates = [x for x in time_rates for _ in (0, 1)]  # double each entry
        times, rates = zip(*time_rates)

        # add on entry in front and back for non interpolated curve
        rates = [rates[0]] + list(rates)
        times = list(times) + [times[-1]]
    else:
        times, rates = zip(*time_rates)

    # plot the binned spikes and the optional curve
    plot_histogram_and_target_curve(bins, times, rates, plot_target)

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
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], bool(sys.argv[3]))
    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2],
             evaluate_string_true(sys.argv[3]), 
             evaluate_string_true(sys.argv[4]))
    else:
        usage()
        exit(1)
