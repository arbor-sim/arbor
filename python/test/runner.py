# -*- coding: utf-8 -*-
#
# runner.py

import unittest

import options

from unit import test_contexts
from unit_distributed import test_contexts_arbmpi
from unit_distributed import test_contexts_mpi4py
# add more if needed

test_modules = [\
    test_contexts,\
    test_contexts_arbmpi,\
    test_contexts_mpi4py\
] # add more if needed

"""
suite
    Goal:    add all tests
    Returns: suite of all tests
"""
def suite(): 
    loader = unittest.TestLoader()

    suites = []
    for test_module in test_modules: 
        test_module_suite = test_module.suite()
        suites.append(test_module_suite)

    suite = unittest.TestSuite(suites)

    return suite
    
if __name__ == "__main__": 
    v = options.parse_arguments().verbosity
    runner = unittest.TextTestRunner(verbosity = v)
    runner.run(suite())
