# -*- coding: utf-8 -*-
#
# runner.py

import unittest

import options

from unit import test_contexts
from unit_distributed import test_contexts_arbmpi
from unit_distributed import test_contexts_mpi4py
# add more if needed

"""
suite
    Goal:    add all tests
    Returns: suite of all tests
"""
def suite(): 

    suite = unittest.TestSuite()

    suite.addTest(test_contexts.suite())
    suite.addTest(test_contexts_arbmpi.suite())
    suite.addTest(test_contexts_mpi4py.suite())
    # add more if needed

    return suite

if __name__ == "__main__": 
    
    runner = unittest.TextTestRunner(verbosity = options.verbosity)
    runner.run(suite())
