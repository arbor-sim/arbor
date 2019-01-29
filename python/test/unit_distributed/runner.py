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

test_modules = [\
    test_contexts_arbmpi,\
    test_contexts_mpi4py\
] # add more if needed

"""
suite
    Goal:    add all tests in this directory
    Returns: suite of tests in this directory
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
