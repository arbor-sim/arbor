# -*- coding: utf-8 -*-
#
# runner.py

import unittest
import arbor as arb

# check Arbor's configuration of mpi
mpi_enabled    = arb.__config__["mpi"]
mpi4py_enabled = arb.__config__["mpi4py"]

if (mpi_enabled and mpi4py_enabled):
    import mpi4py.MPI as mpi

# to be able to run .py file from child directory
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    import options
    import test_contexts_arbmpi
    import test_contexts_mpi4py
    import test_domain_decompositions
    # add more if needed
except ModuleNotFoundError:
    from test import options
    from test.unit_distributed import test_contexts_arbmpi
    from test.unit_distributed import test_contexts_mpi4py
    from test.unit_distributed import test_domain_decompositions
    # add more if needed

test_modules = [\
    test_contexts_arbmpi,\
    test_contexts_mpi4py,\
    test_domain_decompositions\
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

    if not arb.mpi_is_initialized():
        print(" Runner initializing mpi")
        arb.mpi_init()

    if mpi4py_enabled:
        comm = arb.mpi_comm(mpi.COMM_WORLD)
    elif mpi_enabled:
        comm = arb.mpi_comm()

    alloc = arb.proc_allocation()
    ctx = arb.context(alloc, comm)
    rank = ctx.rank

    if rank == 0:
        runner = unittest.TextTestRunner(verbosity = v)
    else:
        sys.stdout = open(os.devnull, 'w')
        runner = unittest.TextTestRunner(stream=sys.stdout)

    runner.run(suite())

    if not arb.mpi_is_finalized():
       arb.mpi_finalize()
