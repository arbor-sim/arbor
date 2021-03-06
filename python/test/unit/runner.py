# -*- coding: utf-8 -*-
#
# runner.py

import unittest

# to be able to run .py file from child directory
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    import options
    import test_cable_probes
    import test_catalogues
    import test_contexts
    import test_decor
    import test_domain_decomposition
    import test_event_generators
    import test_identifiers
    import test_morphology
    import test_schedules
    import test_spikes
    import test_tests
    # add more if needed
except ModuleNotFoundError:
    from test import options
    from test.unit import test_cable_probes
    from test.unit import test_catalogues
    from test.unit import test_contexts
    from test.unit import test_decor
    from test.unit import test_domain_decompositions
    from test.unit import test_event_generators
    from test.unit import test_identifiers
    from test.unit import test_morphology
    from test.unit import test_schedules
    from test.unit import test_spikes
    # add more if needed

test_modules = [\
    test_cable_probes,\
    test_catalogues,\
    test_contexts,\
    test_decor,\
    test_domain_decompositions,\
    test_event_generators,\
    test_identifiers,\
    test_morphology,\
    test_schedules,\
    test_spikes,\
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
    result = runner.run(suite())
    sys.exit(not(result.wasSuccessful()))
