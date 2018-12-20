# -*- coding: utf-8 -*-
#
# options.py

import argparse

import os

"""
parse_arguments
    Goal:       define and read arguments given by the user
    Arguments:  args      - list of strings to parse, default from sys.argv
                namespace - object to take the attributes, default new empty namespace object
    Returns:    attributes of namespace
    Example:    args = parse_arguments() 
"""
def parse_arguments(args=None, namespace=None):
    parser = argparse.ArgumentParser()

    # add arguments as needed (e.g. -d, --dryrun Number of dry run ranks)
    parser.add_argument("-v", "--verbosity", nargs='?', const=2, type=int, choices=[0, 1, 2], default=2, help="increase output verbosity")
    args = parser.parse_args()
    return args

"""
bool_env
    Goal:       get an environment variable coerced to a boolean value.
    Arguments:  var_name  - name of the environment variable
                default   - default to use if `var_name` is not specified in environment
    Returns:    `var_name` or `default` coerced to a boolean using the following rules
                False: "False", "false", "OFF", "off", "Off", ""; True: any other non-empty string
    Example:
                Bash:
                $ export SOME_VAL=True
                Python:
                SOME_VAL = bool_env('SOME_VAL', False)
"""
def bool_env(var_name, default=False):
    test_val = os.getenv(var_name, default)
    # Explicitly check for 'False', 'false','OFF', 'Off', 'off', and '0' since all non-empty
    # string are normally coerced to True.
    if test_val in ('False', 'false', 'OFF', 'Off', 'off', '0'):
        return False
    return bool(test_val) 

"""
========================= Global options ======================== 
read and set command line arguments
"""
args = parse_arguments()
verbosity = args.verbosity
print("Verbosity is set to", verbosity)
# include more if needed

"""
read and set environment variables
"""
TEST_MPI    =  bool_env('ARB_MPI_ENABLED', False)
TEST_MPI4PY =  bool_env('ARB_WITH_MPI4PY', False)
print("ARB_MPI_ENABLED is", TEST_MPI)
print("ARB_WITH_MPI4PY is", TEST_MPI4PY)
# include more if needed
