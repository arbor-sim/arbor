# -*- coding: utf-8 -*-
#
# test_morphology.py

import unittest
import arbor as A
import numpy as N
import math

"""
tests for morphology-related classes
"""


def as_matrix(iso):
    trans = N.array(iso((0, 0, 0)))
    return N.c_[
        N.array([iso(v) for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]).transpose()
        - N.c_[trans, trans, trans],
        trans,
    ]


class TestPlacePwlin(unittest.TestCase):
    def test_identity(self):
        self.assertTrue(N.isclose(as_matrix(A.isometry()), N.eye(3, 4)).all())

    def test_translation(self):
        displacement = (4, 5, 6)
        iso = A.isometry.translate(displacement)
        expected = N.c_[N.eye(3), displacement]
        self.assertTrue(N.isclose(as_matrix(iso), expected).all())

    def test_rotation(self):
        # 90 degrees about y axis.
        iso = A.isometry.rotate(theta=math.pi / 2, axis=(0, 1, 0))
        expected = N.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
        self.assertTrue(N.isclose(as_matrix(iso), expected).all())

    def test_compose(self):
        # Translations are always extrinsic, rotations are intrinsic.
        y90 = A.isometry.rotate(theta=math.pi / 2, axis=(0, 1, 0))
        z90 = A.isometry.rotate(theta=math.pi / 2, axis=(0, 0, 1))
        t123 = A.isometry.translate(1, 2, 3)
        t456 = A.isometry.translate(4, 5, 6)
        iso = (
            t123 * z90 * t456 * y90
        )  # rot about y, then translate, then rot about z, then translate.
        expected = N.array([[0, 0, 1, 5], [1, 0, 0, 7], [0, 1, 0, 9]])
        self.assertTrue(N.isclose(as_matrix(iso), expected).all())

    def test_mpoint(self):
        # Translations can be built from mpoints, and isometry can act upon mpoints. Radius is ignored.
        y90 = A.isometry.rotate(theta=math.pi / 2, axis=(0, 1, 0))
        z90 = A.isometry.rotate(theta=math.pi / 2, axis=(0, 0, 1))
        t123 = A.isometry.translate(A.mpoint(1, 2, 3, 20))
        t456 = A.isometry.translate(A.mpoint(4, 5, 6, 30))
        iso = t123 * z90 * t456 * y90
        expected = N.array([[0, 0, 1, 5], [1, 0, 0, 7], [0, 1, 0, 9]])
        self.assertTrue(N.isclose(as_matrix(iso), expected).all())

        q = iso(A.mpoint(2, 3, 4, 10))
        q_arr = N.array((q.x, q.y, q.z, q.radius))
        q_expected = N.array([4 + 5, 2 + 7, 3 + 9, 10])
        self.assertTrue(N.isclose(q_arr, q_expected).all())

    def test_place_pwlin_id(self):
        # Single branch, discontiguous segments.
        s0p = A.mpoint(0, 0, 0, 10)
        s0d = A.mpoint(1, 0, 0, 10)
        s1p = A.mpoint(3, 0, 0, 10)
        s1d = A.mpoint(4, 0, 0, 10)

        tree = A.segment_tree()
        i = tree.append(A.mnpos, s0p, s0d, 1)
        tree.append(i, s1p, s1d, 2)

        m = A.morphology(tree)
        place = A.place_pwlin(m)

        L0 = place.at(A.location(0, 0))
        L0s = place.all_at(A.location(0, 0))
        self.assertEqual(s0p, L0)
        self.assertEqual([s0p], L0s)

        Lhalf = place.at(A.location(0, 0.5))
        Lhalfs = place.all_at(A.location(0, 0.5))
        self.assertTrue(s0d == Lhalf or s1p == Lhalf)
        self.assertTrue([s0d, s1p] == Lhalfs)

        Chalf = [(s.prox, s.dist) for s in place.segments([A.cable(0, 0.0, 0.5)])]
        self.assertEqual([(s0p, s0d)], Chalf)

        Chalf_all = [
            (s.prox, s.dist) for s in place.all_segments([A.cable(0, 0.0, 0.5)])
        ]
        self.assertEqual([(s0p, s0d), (s1p, s1p)], Chalf_all)

    def test_place_pwlin_isometry(self):
        # Single branch, discontiguous segments.
        s0p = A.mpoint(0, 0, 0, 10)
        s0d = A.mpoint(1, 0, 0, 10)
        s1p = A.mpoint(3, 0, 0, 10)
        s1d = A.mpoint(4, 0, 0, 10)

        tree = A.segment_tree()
        i = tree.append(A.mnpos, s0p, s0d, 1)
        tree.append(i, s1p, s1d, 2)

        m = A.morphology(tree)
        iso = A.isometry.translate(2, 3, 4)
        place = A.place_pwlin(m, iso)

        x0p = iso(s0p)
        x0d = iso(s0d)
        x1p = iso(s1p)

        L0 = place.at(A.location(0, 0))
        L0s = place.all_at(A.location(0, 0))
        self.assertEqual(x0p, L0)
        self.assertEqual([x0p], L0s)

        Lhalf = place.at(A.location(0, 0.5))
        Lhalfs = place.all_at(A.location(0, 0.5))
        self.assertTrue(x0d == Lhalf or x1p == Lhalf)
        self.assertTrue([x0d, x1p] == Lhalfs)

        Chalf = [(s.prox, s.dist) for s in place.segments([A.cable(0, 0.0, 0.5)])]
        self.assertEqual([(x0p, x0d)], Chalf)

        Chalf_all = [
            (s.prox, s.dist) for s in place.all_segments([A.cable(0, 0.0, 0.5)])
        ]
        self.assertEqual([(x0p, x0d), (x1p, x1p)], Chalf_all)
