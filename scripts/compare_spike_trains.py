#!/usr/bin/env python3
import os
import sys
import collections
from itertools import zip_longest

def usage(default_delta):
    print("""
    compare two input spike time files on equality with a delta of max_delta
    order of the spikes is not important. GID should always be equal
    Display the first 50 differences encountered.

    python compare.py file1 file2 (delta={0})

    Produces the file differences.txt with all the found difference in a
    ; seperated format, a single difference per line


    """.format(default_delta))

def parse_file(path):
    """
    Parse a spikes file: fail hard! If parsing does not work,
    print offending line and exit(1)

    returns dict keyed on gid.
        each value is a list of (spiketimes,  the line number, line )
    """
    fp = open(path, "r")
    parsed_data = collections.defaultdict(list)

    line_idx = 0
    for line in fp.readlines():
        stripped_line =  line.strip()
        split_items = stripped_line.split()
        try:
            gid = int(split_items[0].strip())
            time = float(split_items[1].strip())
        except:
            print("Could not parse a line in the file!!!! \n")
            print(" line: " , line_idx, ": ", stripped_line)
            print(path)
            
            exit(1) #failure

        line_data = (line_idx, time, stripped_line)
        parsed_data[gid].append(line_data)
        line_idx += 1

    return parsed_data


def compare(path1, data1, path2, data2, delta):
    """
    compares data1 and data2 on spike time equality.
    if a problem is found at the delta level, store result
    if whole spikes are missing just exit!

    print all errors and then exit(1)
    """
    combined_data = collections.defaultdict(lambda : [[],[]])
    
    for gid, spike_data in list(data1.items()):
        combined_data[gid][0].extend(spike_data)


    for gid, spike_data in list(data2.items()):
        combined_data[gid][1].extend(spike_data)

    different_spikes = []
    for gid, (data_1, data_2)in list(combined_data.items()):
        gid_list1 = data_1
        gid_list2 = data_2

        if len(gid_list1) != len(gid_list2):
            for idx, (time1, time2) in enumerate(zip_longest(gid_list1, gid_list2)):
                # We have to loop all spikes, check here if we have missing spikes 
                # and treat those different
                if time1 == None or time2 == None:
                    time1 =  "Spike not in file" if time1 == None else time1 
                    time2 =  "Spike not in file" if time2 == None else time2
                    different_spikes.append((gid, time1, time2))
                    continue

                # Do an delta test if we have spikes in both lists.
                if abs(time1[1] - time2[1]) > delta:               
                    different_spikes.append((gid, time1, time2))
                    
            continue

        for  time1, time2 in zip( gid_list1, gid_list2):
            if abs(time1[1] - time2[1]) > delta:
                
                different_spikes.append((gid, time1, time2))

    if len(different_spikes) != 0:
        print("Found difference in spike times, displaying first 50 \n")
        print("key == (line_nr, spike_time, content line parsed)\n")
        print("difference #, gid :  target output !=  simulation output")

        for idx, (gid, time1, time2) in enumerate(different_spikes):
            if idx == 50:
                break

            dif_str = "difference #{0}, {3}: {1} !=  {2}".format(idx, time1, time2, gid)
            print(dif_str)

        print("\n\n")


        # Also output to file (could be done in previous loop, but seperation
        # of concerns)
        fp = open("differences.txt", "w")
        fp.write("# difference index, gid, target output, simulation output\n")
        for idx, (gid, time1, time2) in enumerate(different_spikes):
        
            dif_str = "{0}; {3}; {1}; {2}\n".format(idx, time1, time2, gid)
            fp.write(dif_str) 

        # exit with fault code!
        exit(1)
               
    # we compared the file, no errors found. Exit with 0 code!
    exit(0)

if __name__ == "__main__":

    default_delta = 0.0001
    if len(sys.argv) < 4:
        usage(default_delta)
        exit(1) # failure!

    # We are not doing input validation
    path1= sys.argv[1]
    path2= sys.argv[2]
    data = parse_file(path1)
    data2 = parse_file(path2)

    if (len(sys.argv) == 4):
        delta = float(sys.argv[3])
    else:
        delta = default_delta

    compare(path1, data, path2, data2, delta )
