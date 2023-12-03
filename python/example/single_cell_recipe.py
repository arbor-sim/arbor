#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor as A
from arbor import units as U
import pandas as pd  # You may have to pip install these.
import seaborn as sns  # You may have to pip install these.

# The corresponding generic recipe version of `single_cell_model.py`.

# (1) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm
tree = A.segment_tree()
tree.append(A.mnpos, A.mpoint(-3, 0, 0, 3), A.mpoint(3, 0, 0, 3), tag=1)

# (2) Define the soma and its midpoint
labels = A.label_dict({"soma": "(tag 1)", "midpoint": "(location 0 0.5)"})

# (3) Create cell and set properties
decor = (
    A.decor()
    .set_property(Vm=-40 * U.mV)
    .paint('"soma"', A.density("hh"))
    .place('"midpoint"', A.iclamp(10 * U.ms, 2 * U.ms, 0.8 * U.nA), "iclamp")
    .place('"midpoint"', A.threshold_detector(-10 * U.mV), "detector")
)

cell = A.cable_cell(tree, decor, labels)


# (4) Define a recipe for a single cell and set of probes upon it.
# This constitutes the corresponding generic recipe version of
# `single_cell_model.py`.
class single_recipe(A.recipe):
    # (4.1) The base class constructor must be called first, to ensure that
    # all memory in the wrapped C++ class is initialized correctly.
    def __init__(self):
        A.recipe.__init__(self)
        self.the_props = A.neuron_cable_properties()

    # (4.2) Override the num_cells method
    def num_cells(self):
        return 1

    # (4.3) Override the cell_kind method
    def cell_kind(self, _):
        return A.cell_kind.cable

    # (4.4) Override the cell_description method
    def cell_description(self, gid):
        return cell

    # (4.5) Override the probes method with a voltage probe located on "midpoint"
    def probes(self, _):
        return [A.cable_probe_membrane_voltage('"midpoint"', "Um")]

    # (4.6) Override the global_properties method
    def global_properties(self, kind):
        return self.the_props


# (5) Instantiate recipe.
recipe = single_recipe()

# (6) Create simulation. When their defaults are sufficient, context and domain
# decomposition don't have to be manually specified and the simulation can be
# created with just the recipe as argument.
sim = A.simulation(recipe)

# (7) Create and run simulation and set up 10 kHz (every 0.1 ms) sampling on the
# probe. The probe is located on cell 0, and is the 0th probe on that cell, thus
# has probeset_id (0, 0).
sim.record(A.spike_recording.all)
handle = sim.sample((0, "Um"), A.regular_schedule(0.1 * U.ms))
sim.run(tfinal=30 * U.ms)

# (8) Collect results.
spikes = sim.spikes()
data, meta = sim.samples(handle)[0]

if len(spikes) > 0:
    print("{} spikes:".format(len(spikes)))
    for t in spikes["time"]:
        print(f" * {t:3.3f} ms")
else:
    print("no spikes")

print("Plotting results ...")

df = pd.DataFrame({"t/ms": data[:, 0], "U/mV": data[:, 1]})
sns.relplot(data=df, kind="line", x="t/ms", y="U/mV", errorbar=None).savefig(
    "single_cell_recipe_result.svg"
)

df.to_csv("single_cell_recipe_result.csv", float_format="%g")
