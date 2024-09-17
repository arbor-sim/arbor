import unittest

import arbor as A

from pathlib import Path
from tempfile import TemporaryDirectory as TD
from io import StringIO


acc = """(arbor-component
  (meta-data
    (version "0.9-dev"))
  (cable-cell
    (morphology
      (branch 0 -1
        (segment 0
          (point -3.000000 0.000000 0.000000 3.000000)
          (point 3.000000 0.000000 0.000000 3.000000)
          1)))
    (label-dict
      (region-def "soma"
        (tag 1))
      (locset-def "mid"
        (location 0 0.5)))
    (decor
      (default
        (membrane-potential -40.000000 (scalar 1)))
      (default
        (ion-internal-concentration "ca" 0.000050 (scalar 1)))
      (default
        (ion-external-concentration "ca" 2.000000 (scalar 1)))
      (default
        (ion-reversal-potential "ca" 132.457934 (scalar 1)))
      (default
        (ion-internal-concentration "k" 54.400000 (scalar 1)))
      (default
        (ion-external-concentration "k" 2.500000 (scalar 1)))
      (default
        (ion-reversal-potential "k" -77.000000 (scalar 1)))
      (default
        (ion-internal-concentration "na" 10.000000 (scalar 1)))
      (default
        (ion-external-concentration "na" 140.000000 (scalar 1)))
      (default
        (ion-reversal-potential "na" 50.000000 (scalar 1)))
      (paint
        (tag 1)
        (density
          (mechanism "default::hh"
            ("gnabar" 0.120000)
            ("el" -54.300000)
            ("q10" 0.000000)
            ("gl" 0.000300)
            ("gkbar" 0.036000))))
      (place
        (location 0 0.5)
        (current-clamp
          (envelope
            (10.000000 0.800000)
            (12.000000 0.000000))
          0.000000 0.000000)
        "I Clamp 0"))))
"""


swc_arbor = """1 1 -5.0 0.0 0.0 5.0 -1
2 1 0.0 0.0 0.0 5.0 1
3 1 5.0 0.0 0.0 5.0 2
"""


swc_neuron = """1 1 0.1 0.2 0.3 0.4 -1
"""


asc = """((CellBody)\
 (0 0 0 4)\
)\
((Dendrite)\
 (0 2 0 2)\
 (0 5 0 2)\
 (\
  (-5 5 0 2)\
  |\
  (6 5 0 2)\
 )\
)\
((Axon)\
 (0 -2 0 2)\
 (0 -5 0 2)\
 (\
  (-5 -5 0 2)\
  |\
  (6 -5 0 2)\
 )\
)
"""


def load_string(loaders, morph_str):
    for loader in loaders:
        sio = StringIO(morph_str)
        loader(sio)


def load_file(loaders, morph_str, morph_fn):
    for loader in loaders:
        with TD() as tmp:
            tmp = Path(tmp)
            with open(tmp / morph_fn, "w") as fd:
                fd.write(morph_str)
            with open(tmp / morph_fn) as fd:
                loader(fd)


def load_name(loaders, morph_str, morph_fn):
    for loader in loaders:
        with TD() as tmp:
            tmp = Path(tmp)
            with open(tmp / morph_fn, "w") as fd:
                fd.write(morph_str)
            loader(str(tmp / morph_fn))


def load_pathio(loaders, morph_str, morph_fn):
    for loader in loaders:
        with TD() as tmp:
            tmp = Path(tmp)
            with open(tmp / morph_fn, "w") as fd:
                fd.write(morph_str)
            loader(tmp / morph_fn)


class TestAccIo(unittest.TestCase):
    @staticmethod
    def loaders():
        return (A.load_component,)

    def test_stringio(self):
        load_string(self.loaders(), acc)

    def test_fileio(self):
        load_file(self.loaders(), acc, "test.acc")

    def test_nameio(self):
        load_name(self.loaders(), acc, "test.acc")

    def test_pathio(self):
        load_pathio(self.loaders(), acc, "test.acc")


class TestSwcArborIo(unittest.TestCase):
    @staticmethod
    def loaders():
        return (
            lambda f: A.load_swc_arbor(f).morphology,
            lambda f: A.load_swc_arbor(f).segment_tree,
        )

    def test_stringio(self):
        load_string(self.loaders(), swc_arbor)

    def test_fileio(self):
        load_file(self.loaders(), swc_arbor, "test.swc")

    def test_nameio(self):
        load_name(self.loaders(), swc_arbor, "test.swc")

    def test_pathio(self):
        load_pathio(self.loaders(), swc_arbor, "test.swc")


class TestSwcNeuronIo(unittest.TestCase):
    @staticmethod
    def loaders():
        return (
            lambda f: A.load_swc_neuron(f).morphology,
            lambda f: A.load_swc_neuron(f).segment_tree,
        )

    def test_stringio(self):
        load_string(self.loaders(), swc_neuron)

    def test_fileio(self):
        load_file(self.loaders(), swc_neuron, "test.swc")

    def test_nameio(self):
        load_name(self.loaders(), swc_neuron, "test.swc")

    def test_pathio(self):
        load_pathio(self.loaders(), swc_neuron, "test.swc")


class TestAscIo(unittest.TestCase):
    @staticmethod
    def loaders():
        return (
            lambda f: A.load_asc(f).morphology,
            lambda f: A.load_asc(f).segment_tree,
        )

    def test_stringio(self):
        load_string(self.loaders(), asc)

    def test_fileio(self):
        load_file(self.loaders(), asc, "test.asc")

    def test_nameio(self):
        load_name(self.loaders(), asc, "test.asc")

    def test_pathio(self):
        load_pathio(self.loaders(), asc, "test.asc")


class serdes_recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        self.the_props = A.neuron_cable_properties()
        self.the_props.catalogue = A.default_catalogue()

    def num_cells(self):
        return 1

    def cell_kind(self, _):
        return A.cell_kind.cable

    def cell_description(self, _):
        tree = A.segment_tree()
        s = tree.append(A.mnpos, A.mpoint(-3, 0, 0, 3), A.mpoint(3, 0, 0, 3), tag=1)
        _ = tree.append(s, A.mpoint(3, 0, 0, 1), A.mpoint(33, 0, 0, 1), tag=3)

        dec = A.decor()
        dec.paint("(all)", A.density("pas"))
        dec.discretization(A.cv_policy("(max-extent 1)"))

        return A.cable_cell(tree, dec)

    def global_properties(self, _):
        return self.the_props


# Very simple test for SerDes
class TestSerdes(unittest.TestCase):
    def test_serialize(self):
        self.maxDiff = 1024 * 1024
        rec = serdes_recipe()
        sim = A.simulation(rec)
        jsn = sim.serialize()
        try:
            sim.deserialize(jsn)
        except RuntimeError as e:
            print(f"Unexpected error\n{e}\nin JSON:\n{jsn}")
            raise
