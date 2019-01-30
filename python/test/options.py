# -*- coding: utf-8 -*-
#
# options.py

import argparse

import os

def parse_arguments(args=None, namespace=None):
    parser = argparse.ArgumentParser()

    # add arguments as needed (e.g. -d, --dryrun Number of dry run ranks)
    parser.add_argument("-v", "--verbosity", nargs='?', const=0, type=int, choices=[0, 1, 2], default=0, help="increase output verbosity")
    #parser.add_argument("-d", "--dryrun", type=int, default=100 , help="number of dry run ranks")
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
read and set environment variables 
(parsing of arguments in TestClasses (e.g. as setUpClass to pass arguments) 
  due to interference with unittest arguments)
"""
TEST_MPI    =  (bool_env('ARB_MPI_ENABLED', False) or bool_env('TEST_MPI', False))
TEST_MPI4PY =  (bool_env('ARB_WITH_MPI4PY', False) or bool_env('TEST_MPI4PY', False))
print("ARB_MPI_ENABLED on?", TEST_MPI)
print("ARB_WITH_MPI4PY on?", TEST_MPI4PY)
# include more if needed
