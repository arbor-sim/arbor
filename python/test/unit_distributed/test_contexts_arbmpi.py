# -*- coding: utf-8 -*-
#
# test_contexts_arbmpi.py

import unittest

import pyarb as arb

# to be able to run .py file from child directory
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    import options
except ModuleNotFoundError:
    from test import options

"""
TestContextArbMPI
   Goal: collect all tests for testing distributed arb.context using arbor mpi wrappers
"""
@unittest.skipIf(options.TEST_MPI == False, "ARB_MPI_ENABLED=OFF")
class TestContextsArbMPI(unittest.TestCase): 
    # Initialize mpi only once in this class (when adding classes move initialization to setUpModule()
    @classmethod
    def setUpClass(self):
        #print("setUp --- TestContextMPI class")
        if(arb.mpi_is_initialized() == False): 
            #print("    Initializing mpi")
            arb.mpi_init()
        #else:
            #print("    mpi already initialized")
    # Finalize mpi only once in this class (when adding classes move finalization to setUpModule()
    @classmethod
    def tearDownClass(self):
        #print("tearDown --- TestContextMPI class")
        #print("    Finalizing mpi")
        if (options.TEST_MPI4PY == False and arb.mpi_is_finalized() == False):
            #print("    Finalizing mpi")
            arb.mpi_finalize()
        #else:
            #print("    No finalizing due to further testing with mpi4py")
    
    def test_initialized_arbmpi(self):
        self.assertTrue(arb.mpi_is_initialized())

    def test_context_arbmpi(self):
        comm = arb.mpi_comm()

        # test that by default communicator is MPI_COMM_WORLD
        self.assertEqual(str(comm), '<mpi_comm: MPI_COMM_WORLD>')
        #print(comm)

        # test context with mpi
        alloc = arb.proc_allocation()
        ctx = arb.context(alloc, comm)

        self.assertEqual(ctx.threads, alloc.threads)
        self.assertTrue(ctx.has_mpi)
        #print(ctx)

    def test_finalized_arbmpi(self):
        self.assertFalse(arb.mpi_is_finalized())

def suite():
    suite = unittest.makeSuite(TestContextsArbMPI, 'test')
    return suite

def run():
    runner = unittest.TextTestRunner(verbosity = options.verbosity)
    runner.run(suite())

if __name__ == "__main__":
    run()
