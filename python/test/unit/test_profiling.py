# -*- coding: utf-8 -*-
#
# test_schedules.py

import unittest

import arbor as A
from arbor import units as U
import functools

"""
all tests for profiling
"""


def lazy_skipIf(condition, reason):
    """
    Postpone skip evaluation until test is ran by evaluating callable `condition`
    """

    def inner_decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            if condition():
                raise unittest.SkipTest(reason)
            else:
                return f(*args, **kwargs)

        return wrapped

    return inner_decorator


class a_recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        self.props = A.neuron_cable_properties()
        self.trains = [[0.8, 2, 2.1, 3], [0.4, 2, 2.2, 3.1, 4.5], [0.2, 2, 2.8, 3]]

    def num_cells(self):
        return 3

    def cell_kind(self, gid):
        return A.cell_kind.spike_source

    def connections_on(self, gid):
        return []

    def event_generators(self, gid):
        return []

    def global_properties(self, kind):
        return self.the_props

    def probes(self, gid):
        return []

    def cell_description(self, gid):
        sched = A.explicit_schedule([t * U.ms for t in self.trains[gid]])
        return A.spike_source_cell("src", sched)


def skipWithoutSupport():
    return not bool(A.config().get("profiling", False))


class TestProfiling(unittest.TestCase):
    def test_support(self):
        self.assertTrue("profiling" in A.config(), "profiling key not in config")
        profiling_support = A.config()["profiling"]
        self.assertEqual(bool, type(profiling_support), "profiling flag should be bool")
        if profiling_support:
            self.assertTrue(
                hasattr(A, "profiler_initialize"),
                "missing profiling interface with profiling support",
            )
            self.assertTrue(
                hasattr(A, "profiler_summary"),
                "missing profiling interface with profiling support",
            )
        else:
            self.assertFalse(
                hasattr(A, "profiler_initialize"),
                "profiling interface without profiling support",
            )
            self.assertFalse(
                hasattr(A, "profiler_summary"),
                "profiling interface without profiling support",
            )

    @lazy_skipIf(skipWithoutSupport, "run test only with profiling support")
    def test_summary(self):
        context = A.context()
        A.profiler_initialize(context)
        recipe = a_recipe()
        dd = A.partition_load_balance(recipe, context)
        A.simulation(recipe, context, dd).run(1 * U.ms)
        summary = A.profiler_summary()
        self.assertEqual(str, type(summary), "profiler summary must be str")
        self.assertTrue(summary, "empty summary")
