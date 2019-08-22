# -*- coding: utf-8 -*-
#
# runner.py

import unittest

# to be able to run .py file from child directory
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    import options
    import test_contexts
    import test_domain_decomposition
    import test_event_generators
    import test_identifiers
    import test_tests
    import test_schedules
    # add more if needed
except ModuleNotFoundError:
    from test import options
    from test.unit import test_contexts
    from test.unit import test_domain_decompositions
    from test.unit import test_event_generators
    from test.unit import test_identifiers
    from test.unit import test_schedules
    # add more if needed

test_modules = [\
    test_contexts,\
    test_domain_decompositions,\
    test_event_generators,\
    test_identifiers,\
    test_schedules\
] # add more if needed

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
