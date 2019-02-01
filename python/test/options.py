# -*- coding: utf-8 -*-
#
# options.py

import argparse

import os

import arbor as arb

def parse_arguments(args=None, namespace=None):
    parser = argparse.ArgumentParser()

    # add arguments as needed (e.g. -d, --dryrun Number of dry run ranks)
    parser.add_argument("-v", "--verbosity", nargs='?', const=0, type=int, choices=[0, 1, 2], default=0, help="increase output verbosity")
    #parser.add_argument("-d", "--dryrun", type=int, default=100 , help="number of dry run ranks")
    args = parser.parse_args()
    return args
