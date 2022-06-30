#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor
import pandas  # You may have to pip install these.
import seaborn  # You may have to pip install these.

# The corresponding generic recipe version of `single_cell_model.py`.

# (1) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm

tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

# (2) Define the soma and its midpoint

labels = arbor.label_dict({"soma": "(tag 1)", "midpoint": "(location 0 0.5)"})

# (3) Create cell and set properties

decor = arbor.decor()
decor.set_property(Vm=-40)
decor.paint('"soma"', arbor.density("hh"))
decor.place('"midpoint"', arbor.iclamp(10, 2, 0.8), "iclamp")
decor.place('"midpoint"', arbor.spike_detector(-10), "detector")
cell = arbor.cable_cell(tree, labels, decor)

# (4) Define a recipe for a single cell and set of probes upon it.
# This constitutes the corresponding generic recipe version of
# `single_cell_model.py`.


class single_recipe(arbor.recipe):
    def __init__(self):
        # (4.1) The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)
        self.the_props = arbor.neuron_cable_properties()

    def num_cells(self):
        # (4.2) Override the num_cells method
        return 1

    def cell_kind(self, gid):
        # (4.3) Override the cell_kind method
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        # (4.4) Override the cell_description method
        return cell

    def probes(self, gid):
        # (4.5) Override the probes method with a voltage probe located on "midpoint"
        return [arbor.cable_probe_membrane_voltage('"midpoint"')]

    def global_properties(self, kind):
        # (4.6) Override the global_properties method
        return self.the_props


# (5) Instantiate recipe.

recipe = single_recipe()

# (6) Create simulation. When their defaults are sufficient, context and domain decomposition don't
# have to be manually specified and the simulation can be created with just the recipe as argument.

sim = arbor.simulation(recipe)

# (7) Create and run simulation and set up 10 kHz (every 0.1 ms) sampling on the probe.
# The probe is located on cell 0, and is the 0th probe on that cell, thus has probe_id (0, 0).

sim.record(arbor.spike_recording.all)
handle = sim.sample((0, 0), arbor.regular_schedule(0.1))
sim.run(tfinal=30)

# (8) Collect results.

spikes = sim.spikes()
data, meta = sim.samples(handle)[0]

if len(spikes) > 0:
    print("{} spikes:".format(len(spikes)))
    for t in spikes["time"]:
        print("{:3.3f}".format(t))
else:
    print("no spikes")

print("Plotting results ...")

df = pandas.DataFrame({"t/ms": data[:, 0], "U/mV": data[:, 1]})
seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV", ci=None).savefig(
    "single_cell_recipe_result.svg"
)

df.to_csv("single_cell_recipe_result.csv", float_format="%g")
