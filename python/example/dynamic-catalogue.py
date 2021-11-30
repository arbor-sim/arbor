#!/usr/bin/env python3

from pathlib import Path

import arbor as arb

cat = 'cat-catalogue.so'

class recipe(arb.recipe):
    def __init__(self):
        arb.recipe.__init__(self)
        self.tree = arb.segment_tree()
        self.tree.append(arb.mnpos, (0, 0, 0, 10), (1, 0, 0, 10), 1)
        self.props = arb.neuron_cable_properties()
        self.cat = arb.load_catalogue(cat)
        self.props.register(self.cat)
        d = arb.decor()
        d.paint('(all)', 'dummy')
        d.set_property(Vm=0.0)
        self.cell = arb.cable_cell(self.tree, arb.label_dict(), d)

    def global_properties(self, _):
        return self.props

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return arb.cell_kind.cable

    def cell_description(self, gid):
        return self.cell

if not Path(cat).is_file():
    print("""Catalogue not found in this directory.
Please ensure it has been compiled by calling")
  <arbor>/scripts/build-catalogue cat <arbor>/python/examples/cat
where <arbor> is the location of the arbor source tree.""")
    exit(1)

rcp = recipe()
ctx = arb.context()
dom = arb.partition_load_balance(rcp, ctx)
sim = arb.simulation(rcp, dom, ctx)
sim.run(tfinal=30)
