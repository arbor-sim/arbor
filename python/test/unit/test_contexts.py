# -*- coding: utf-8 -*-
#
# test_contexts.py

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
TestContext
    Goal: collect all tests for testing non-distributed arb.context 
"""

class TestContexts(unittest.TestCase):
    def test_default(self):
        ctx = arb.context()

    def test_resources(self):
        alloc = arb.proc_allocation()

        # test that by default proc_allocation has 1 thread and no GPU, no MPI
        self.assertEqual(alloc.threads, 1)
        self.assertFalse(alloc.has_gpu)
        self.assertEqual(alloc.gpu_id, -1)

        alloc.threads = 20
        self.assertEqual(alloc.threads, 20)

    def test_context(self):
        alloc = arb.proc_allocation()

        ctx1 = arb.context()

        self.assertEqual(ctx1.threads, alloc.threads)
        self.assertEqual(ctx1.has_gpu, alloc.has_gpu)

        # default construction does not use GPU or MPI
        self.assertEqual(ctx1.threads, 1)        
        self.assertFalse(ctx1.has_gpu)
        self.assertFalse(ctx1.has_mpi)
        self.assertEqual(ctx1.ranks, 1)
        self.assertEqual(ctx1.rank, 0)
        #print(ctx1)

        # change allocation
        alloc.threads = 23
        self.assertEqual(alloc.threads, 23)
        alloc.gpu_id = -1
        self.assertEqual(alloc.gpu_id, -1)
        #print(alloc)

        # test context construction with proc_allocation()
        ctx2 = arb.context(alloc)
        self.assertEqual(ctx2.threads, alloc.threads)
        self.assertEqual(ctx2.has_gpu, alloc.has_gpu)
        self.assertEqual(ctx2.ranks, 1)
        self.assertEqual(ctx2.rank, 0)       
        #print(ctx2)
      
def suite():
    suite = unittest.makeSuite(TestContexts, 'test')
    return suite

def run():
    runner = unittest.TextTestRunner(verbosity = options.verbosity)
    runner.run(suite())

if __name__ == "__main__":
    run()
