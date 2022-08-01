#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor
import pandas
import seaborn
import sys
from arbor import density

# (1) Read the morphology from an SWC file.

# Read the SWC filename from input
# Example from docs: single_cell_detailed.swc

if len(sys.argv) < 2:
    print("No SWC file passed to the program")
    sys.exit(0)

filename = sys.argv[1]
morph = arbor.load_swc_arbor(filename)

# (2) Create and populate the label dictionary.

# Label dict, with Pre-defined labels soma, axon, dend, and apic
labels = arbor.label_dict().add_swc_tags()

# Regions:

# Add a label for a region that includes the whole morphology
labels["all"] = "(all)"
# Add a label for the parts of the morphology with radius greater than 1.5 μm.
labels["gt_1.5"] = '(radius-ge (region "all") 1.5)'
# Join regions "apic" and "gt_1.5"
labels["custom"] = '(join (region "apic") (region "gt_1.5"))'

# Locsets:

# Add a labels for the root of the morphology and all the terminal points
labels["root"] = "(root)"
labels["terminal"] = "(terminal)"
# Add a label for the terminal locations in the "custom" region:
labels["custom_terminal"] = '(restrict (locset "terminal") (region "custom"))'
# Add a label for the terminal locations in the "axon" region:
labels["axon_terminal"] = '(restrict (locset "terminal") (region "axon"))'

# (3) Create and populate the decor.

decor = arbor.decor()

# Set the default properties of the cell (this overrides the model defaults).
decor.set_property(Vm=-55)
decor.set_ion("na", int_con=10, ext_con=140, rev_pot=50, method="nernst/na")
decor.set_ion("k", int_con=54.4, ext_con=2.5, rev_pot=-77)

# Override the cell defaults.
decor.paint('"custom"', tempK=270)
decor.paint('"soma"', Vm=-50)

# Paint density mechanisms.
decor.paint('"all"', density("pas"))
decor.paint('"custom"', density("hh"))
decor.paint('"dend"', density("Ih", {"gbar": 0.001}))

# Place stimuli and spike detectors.
decor.place('"root"', arbor.iclamp(10, 1, current=2), "iclamp0")
decor.place('"root"', arbor.iclamp(30, 1, current=2), "iclamp1")
decor.place('"root"', arbor.iclamp(50, 1, current=2), "iclamp2")
decor.place('"axon_terminal"', arbor.spike_detector(-10), "detector")

# Single CV for the "soma" region
soma_policy = arbor.cv_policy_single('"soma"')
# Single CV for the "soma" region
dflt_policy = arbor.cv_policy_max_extent(1.0)
# default policy everywhere except the soma
policy = dflt_policy | soma_policy
# Set cv_policy
decor.discretization(policy)

# (4) Create the cell.

cell = arbor.cable_cell(morph, labels, decor)


# (5) Create a class that inherits from arbor.recipe
class single_recipe(arbor.recipe):

    # (5.1) Define the class constructor
    def __init__(self):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)

        self.the_props = arbor.cable_global_properties()
        self.the_props.set_property(Vm=-65, tempK=300, rL=35.4, cm=0.01)
        self.the_props.set_ion(
            ion="na", int_con=10, ext_con=140, rev_pot=50, method="nernst/na"
        )
        self.the_props.set_ion(ion="k", int_con=54.4, ext_con=2.5, rev_pot=-77)
        self.the_props.set_ion(ion="ca", int_con=5e-5, ext_con=2, rev_pot=132.5)
        self.the_props.catalogue.extend(arbor.allen_catalogue(), "")

    # (5.2) Override the num_cells method
    def num_cells(self):
        return 1

    # (5.3) Override the cell_kind method
    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    # (5.4) Override the cell_description method
    def cell_description(self, gid):
        return cell

    # (5.5) Override the probes method
    def probes(self, gid):
        return [arbor.cable_probe_membrane_voltage('"custom_terminal"')]

    # (5.6) Override the global_properties method
    def global_properties(self, gid):
        return self.the_props


# Instantiate recipe
recipe = single_recipe()

# (6) Create a simulation
sim = arbor.simulation(recipe)

# Instruct the simulation to record the spikes and sample the probe
sim.record(arbor.spike_recording.all)

probe_id = arbor.cell_member(0, 0)
handle = sim.sample(probe_id, arbor.regular_schedule(0.02))

# (7) Run the simulation
sim.run(tfinal=100, dt=0.025)

# (8) Print or display the results
spikes = sim.spikes()
print(len(spikes), "spikes recorded:")
for s in spikes:
    print(s)

data = []
meta = []
for d, m in sim.samples(handle):
    data.append(d)
    meta.append(m)

df_list = []
for i in range(len(data)):
    df_list.append(
        pandas.DataFrame(
            {
                "t/ms": data[i][:, 0],
                "U/mV": data[i][:, 1],
                "Location": str(meta[i]),
                "Variable": "voltage",
            }
        )
    )
df = pandas.concat(df_list, ignore_index=True)
seaborn.relplot(
    data=df, kind="line", x="t/ms", y="U/mV", hue="Location", col="Variable", ci=None
).savefig("single_cell_recipe_result.svg")
