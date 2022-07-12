# -*- coding: utf-8 -*-
#
# test_schedules.py

import unittest

import arbor as arb
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


class a_recipe(arb.recipe):
    def __init__(self):
        arb.recipe.__init__(self)
        self.props = arb.neuron_cable_properties()
        self.trains = [[0.8, 2, 2.1, 3], [0.4, 2, 2.2, 3.1, 4.5], [0.2, 2, 2.8, 3]]

    def num_cells(self):
        return 3

    def cell_kind(self, gid):
        return arb.cell_kind.spike_source

    def connections_on(self, gid):
        return []

    def event_generators(self, gid):
        return []

    def global_properties(self, kind):
        return self.the_props

    def probes(self, gid):
        return []

    def cell_description(self, gid):
        return arb.spike_source_cell("src", arb.explicit_schedule(self.trains[gid]))


def skipWithoutSupport():
    return not bool(arb.config().get("profiling", False))


class TestProfiling(unittest.TestCase):
    def test_support(self):
        self.assertTrue("profiling" in arb.config(), "profiling key not in config")
        profiling_support = arb.config()["profiling"]
        self.assertEqual(bool, type(profiling_support), "profiling flag should be bool")
        if profiling_support:
            self.assertTrue(
                hasattr(arb, "profiler_initialize"),
                "missing profiling interface with profiling support",
            )
            self.assertTrue(
                hasattr(arb, "profiler_summary"),
                "missing profiling interface with profiling support",
            )
        else:
            self.assertFalse(
                hasattr(arb, "profiler_initialize"),
                "profiling interface without profiling support",
            )
            self.assertFalse(
                hasattr(arb, "profiler_summary"),
                "profiling interface without profiling support",
            )

    @lazy_skipIf(skipWithoutSupport, "run test only with profiling support")
    def test_summary(self):
        context = arb.context()
        arb.profiler_initialize(context)
        recipe = a_recipe()
        dd = arb.partition_load_balance(recipe, context)
        arb.simulation(recipe, context, dd).run(1)
        summary = arb.profiler_summary()
        self.assertEqual(str, type(summary), "profiler summary must be str")
        self.assertTrue(summary, "empty summary")
