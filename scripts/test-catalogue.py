#!/usr/bin/env python3

import argparse
import os
import sys

import arbor

P = argparse.ArgumentParser(
    description="Verify that a mechanism catalogue can be loaded through Python interface."
)
P.add_argument("catname", metavar="FILE", help="path of the catalogue to test.")

args = P.parse_args()
catname = args.catname

print(catname)

if not os.path.isfile(catname):
    print("ERROR: unable to open catalogue file")
    sys.exit(1)

print([n for n in arbor.load_catalogue(catname).keys()])
