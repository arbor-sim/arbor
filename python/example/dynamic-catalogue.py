#!/usr/bin/env python3

from pathlib import Path

import arbor as A
from arbor import units as U

cat = Path("cat-catalogue.so").resolve()


class recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        self.tree = A.segment_tree()
        self.tree.append(A.mnpos, (0, 0, 0, 10), (1, 0, 0, 10), 1)
        self.props = A.neuron_cable_properties()
        self.props.catalogue = A.load_catalogue(cat)
        d = A.decor().paint("(all)", "dummy").set_property(Vm=0.0)
        self.cell = A.cable_cell(self.tree, d)

    def global_properties(self, _):
        return self.props

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        return self.cell


if not cat.is_file():
    print(
        """Catalogue not found in this directory.
Please ensure it has been compiled by calling
  <arbor>/scripts/build-catalogue cat <arbor>/python/example/cat
where <arbor> is the location of the arbor source tree."""
    )
    exit(1)

rcp = recipe()
sim = A.simulation(rcp)
sim.run(tfinal=30 * U.ms)
