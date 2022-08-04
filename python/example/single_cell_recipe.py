#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor
import pandas  # You may have to pip install these.
import seaborn  # You may have to pip install these.

# The corresponding generic recipe version of `single_cell_model.py`.

# (1) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm

tree = arbor.segment_tree()
p = tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)
tree.append(p, arbor.mpoint(3, 0, 0, 3), arbor.mpoint(-3, 0, 0, 3), tag=2)
tree.append(p, arbor.mpoint(3, 0, 0, 3), arbor.mpoint(-3, 0, 0, 3), tag=2)

# (2) Define the soma and its midpoint

labels = arbor.label_dict({"soma": "(tag 1)", "midpoint": "(location 0 0.5)"})

# (3) Create cell and set properties

decor = (
    arbor.decor()
    .set_property(Vm=-40)
    .paint('"soma"', arbor.density("hh"))
    .place('"midpoint"', arbor.iclamp(10, 2, 0.8), "iclamp")
    .place('"midpoint"', arbor.spike_detector(-10), "detector")
)

cell = arbor.cable_cell(tree, labels, decor)

# (4) Define a recipe for a single cell and set of probes upon it.
# This constitutes the corresponding generic recipe version of
# `single_cell_model.py`.


class single_recipe(arbor.recipe):
    # (4.1) The base class constructor must be called first, to ensure that
    # all memory in the wrapped C++ class is initialized correctly.
    def __init__(self):
        arbor.recipe.__init__(self)
        self.the_props = arbor.neuron_cable_properties()

    # (4.2) Override the num_cells method
    def num_cells(self):
        return 1

    # (4.3) Override the cell_kind method
    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    # (4.4) Override the cell_description method
    def cell_description(self, gid):
        return cell

    # (4.5) Override the probes method with a voltage probe located on "midpoint"
    def probes(self, gid):
        return [arbor.cable_probe_membrane_voltage('(location 0 0.5)'),
                arbor.cable_probe_membrane_voltage_cell(),
                arbor.cable_probe_membrane_voltage('(join (location 0 0) (location 0 1))'),
                ]

    # (4.6) Override the global_properties method
    def global_properties(self, kind):
        return self.the_props
recipe = single_recipe()
sim = arbor.simulation(recipe)
handles = [sim.sample((0, n), arbor.regular_schedule(0.1))
           for n in range(3) ]
sim.run(tfinal=1)

for hd in handles:
    print("Handle", hd)
    for d, m in sim.samples(hd):
        print(" * Meta:", m)
        print(" * Payload:", d.shape)
