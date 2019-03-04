# -*- coding: utf-8 -*-
#
# runner.py

import unittest
import arbor as arb

# check Arbor's configuration of mpi
dict = arb.config()
config_mpi = dict["mpi"]
config_mpi4py = dict["mpi4py"]

if (config_mpi and config_mpi4py):
    import mpi4py.MPI as mpi

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

    if config_mpi4py:
        comm = arb.mpi_comm_from_mpi4py(mpi.COMM_WORLD)
    elif config_mpi:
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
        #print(" Runner finalizing mpi")
       arb.mpi_finalize()
    #else:
       #print(" mpi already finalized!")
