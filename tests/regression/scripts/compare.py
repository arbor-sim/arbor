#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import collections

def usage(default_delta):
    print ("""
    compare two input spike time files on equality with a delta of max_delta
    order of the spikes is not important. GID should always be equal

    python compare.py file1 file2 (delta={0})


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
            print ("Could not parse a line in the file!!!! \n")
            print (" line: " , line_idx, ": ", stripped_line)
            print (path)
            
            exit(1) #failure

        line_data = (time, line_idx, stripped_line)
        parsed_data[gid].append(line_data)

    return parsed_data


def compare(path1, data1, path2, data2, delta):
    """
    compares data1 and data2 on spike time equality.
    if a problem is found at the delta level, store result
    if whole spikes are missing just exit!

    print all errors and then exit(1)
    """
    # First check all gids are equal
    if (data1.keys() != data2.keys()):
        print ("The encountered gids are NOT equal!")
        if (len(data1.keys()) == 0):
            print ("No spikes in",  path1)
            exit(1)
            
        if (len(data2.keys()) == 0):
            print ("No spikes in",  path2)
            exit(1)

        # Check the lengths
        if ( len(data1.keys()) > len(data2.keys())):
            print ("Extra gid encountered in:", path1)
            print (set(data1.keys()) - set(data2.keys()) )
            exit(1)
            
        elif ( len(data2.keys()) > len(data1.keys())):
            print ("Extra gid encountered in:", path2)
            print (set(data2.keys()) - set(data1.keys()) )
            exit(1)

        print ("first difference in GIDs in the two files: ")
        for key1, key2 in zip (data1.keys(), data2.keys()):
            if key1 != key2:
                print (key1, " != ", key2)
                exit(1)

        # should not happen but still
        print ("encountered unknown difference in files!")
        exit(1)

    different_spikes = []
    for gid in data1.keys():
        gid_list1 = data1[gid]
        gid_list2 = data2[gid]

        if len(gid_list1) != len(gid_list2):
            print ("Difference in the number of spikes of GID #", gid)
            print ("print first difference!")
            for idx, (time1, time2) in enumerate(map(None, gid_list1, gid_list2)):
                if time1 != time2:
                    print ("Spike #", idx, ": ", time1, " != ", time2)
                    # IF there are ALSO spike time differences these are NOT reported
                    exit(1)

        for  time1, time2 in zip( gid_list1, gid_list2):
            if abs(time1[0] - time2[0]) > delta:
                different_spikes.append((time1, time2))

    if len(different_spikes) != 0:
        print ("Found difference in spike times, displaying first 10:")
        for idx, (time1, time2) in enumerate(different_spikes):
            if idx == 10:
                break

            print ("Spike #", idx, ": ", time1, " != ", time2)
        exit(1)

    # we compared the file, no errors found.
    # exit with 0 code!
    exit(0)

if __name__ == "__main__":

    default_delta = 0.0001
    if len(sys.argv) < 4:
        usage(default_delta)
        exit(1) # failure!
    path1= sys.argv[1]
    path2= sys.argv[2]
    data = parse_file(path1)

    data2 = parse_file(path2)

    if (len(sys.argv) == 4):
        delta = float(sys.argv[3])
    else:
        delta = default_delta

    compare(path1, data, path2, data2, delta )
