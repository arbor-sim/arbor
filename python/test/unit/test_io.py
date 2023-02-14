import unittest

import arbor as A

from pathlib import Path
from tempfile import TemporaryDirectory as TD
from io import StringIO
from functools import partial


acc = """(arbor-component
  (meta-data
    (version "0.1-dev"))
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
        (membrane-potential -40.000000))
      (default
        (ion-internal-concentration "ca" 0.000050))
      (default
        (ion-external-concentration "ca" 2.000000))
      (default
        (ion-reversal-potential "ca" 132.457934))
      (default
        (ion-internal-concentration "k" 54.400000))
      (default
        (ion-external-concentration "k" 2.500000))
      (default
        (ion-reversal-potential "k" -77.000000))
      (default
        (ion-internal-concentration "na" 10.000000))
      (default
        (ion-external-concentration "na" 140.000000))
      (default
        (ion-reversal-potential "na" 50.000000))
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
        return (A.load_swc_arbor, partial(A.load_swc_arbor, raw=True))

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
        return (A.load_swc_neuron, partial(A.load_swc_neuron, raw=True))

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
        return (A.load_asc, partial(A.load_asc, raw=True))

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

    def cell_description(self, gid):
        tree = A.segment_tree()
        s = tree.append(A.mnpos, A.mpoint(-3, 0, 0, 3), A.mpoint(3, 0, 0, 3), tag=1)
        _ = tree.append(s, A.mpoint(3, 0, 0, 1), A.mpoint(33, 0, 0, 1), tag=3)

        dec = A.decor()
        dec.paint("(all)", A.density("pas"))
        dec.discretization(A.cv_policy("(max-extent 1)"))

        return A.cable_cell(tree, dec)

    def global_properties(self, kind):
        return self.the_props

class TestSerdes(unittest.TestCase):

    def test_serialize(self):
        rec = serdes_recipe()
        sim = A.simulation(rec)
        jsn = sim.serialize()
        self.AssertEqual(jsn, {"cell_groups_":[{"gids_":[0],"lowered_":{"seed_":0,"state_":{"cbprng_seed":0,"conductivity":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"deliverable_events":{"ev_data_":[],"ev_time_":[],"mark_":[0],"remaining_":0,"span_begin_":[0],"span_end_":[0]},"dt_cv":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"dt_intdom":[0.0],"ion_data":{},"storage":{"0":{"data_":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.0009999999999999998,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,-70.0],"indices_":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,35,35,35,35],"random_number_update_counter_":0,"random_numbers_":[[],[],[],[]]}},"time":[0.0],"time_since_spike":[],"time_to":[0.0],"voltage":[-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0,-65.0]},"tmin_":0.0},"spikes_":[],"staged_events_":[],"target_handles_":[]}],"epoch_":{"id":-1,"t0":0.0,"t1":0.0},"event_lanes_":[[[]],[[]]],"local_spikes_":[[],[]],"pending_events_":[[]],"t_interval_":8.988465674311579e+307})
        sim.deserialize(jsn)
