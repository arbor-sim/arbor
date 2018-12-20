# -*- coding: utf-8 -*-
#
# runner.py

import unittest

# to be able to run .py file from child directory
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    import options
    import test_contexts_arbmpi
    import test_contexts_mpi4py
    # add more if needed
except ModuleNotFoundError:
    from test import options
    from test.unit_distributed import test_contexts_arbmpi
    from test.unit_distributed import test_contexts_mpi4py
    # add more if needed

"""
suite
    Goal:    add all tests in this directory
    Returns: suite of tests in this directory
"""
def suite(): 

    suite = unittest.TestSuite()

    suite.addTest(test_contexts_arbmpi.suite())
    suite.addTest(test_contexts_mpi4py.suite())
    # add more if needed

    return suite

if __name__ == "__main__": 
    
    runner = unittest.TextTestRunner(verbosity = options.verbosity)
    runner.run(suite())
