#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor as A
from arbor import units as U
import pandas as pd
import seaborn as sns
import sys

# (1) Read the morphology from an SWC file.

# Read the SWC filename from input
# Example from docs: single_cell_detailed.swc

if len(sys.argv) < 2:
    print("No SWC file passed to the program")
    sys.exit(0)

filename = sys.argv[1]
morph = A.load_swc_arbor(filename)

# (2) Create and populate the label dictionary.

labels = A.label_dict(
    {
        # Regions:
        # Add a label for a region that includes the whole morphology
        "all": "(all)",
        # Add a label for the parts of the morphology with radius greater than 1.5 μm.
        "gt_1.5": '(radius-ge (region "all") 1.5)',
        # Join regions "apic" and "gt_1.5"
        "custom": '(join (region "apic") (region "gt_1.5"))',
        # Locsets:
        # Add a labels for the root of the morphology and all the terminal points
        "root": "(root)",
        "terminal": "(terminal)",
        # Add a label for the terminal locations in the "custom" region:
        "custom_terminal": '(restrict-to (locset "terminal") (region "custom"))',
        # Add a label for the terminal locations in the "axon" region:
        "axon_terminal": '(restrict-to (locset "terminal") (region "axon"))',
    }
).add_swc_tags()  # Add SWC pre-defined regions

# (3) Create and populate the decor.

decor = (
    A.decor()
    # Set the default properties of the cell (this overrides the model defaults).
    .set_property(Vm=-55)
    .set_ion("na", int_con=10, ext_con=140, rev_pot=50, method="nernst/na")
    .set_ion("k", int_con=54.4, ext_con=2.5, rev_pot=-77)
    # Override the cell defaults.
    .paint('"custom"', tempK=270)
    .paint('"soma"', Vm=-50)
    # Paint density mechanisms.
    .paint('"all"', A.density("pas"))
    .paint('"custom"', A.density("hh"))
    .paint('"dend"', A.density("Ih", gbar=0.001))
    # Place stimuli and detectors.
    .place('"root"', A.iclamp(10 * U.ms, 1 * U.ms, current=2 * U.nA), "iclamp0")
    .place('"root"', A.iclamp(30 * U.ms, 1 * U.ms, current=2 * U.nA), "iclamp1")
    .place('"root"', A.iclamp(50 * U.ms, 1 * U.ms, current=2 * U.nA), "iclamp2")
    .place('"axon_terminal"', A.threshold_detector(-10 * U.mV), "detector")
    # Set discretisation: Soma as one CV, 1um everywhere else
    .discretization('(replace (single (region "soma")) (max-extent 1.0))')
)


# (4) Create the cell.

cell = A.cable_cell(morph, decor, labels)


# (5) Create a class that inherits from A.recipe
class single_recipe(A.recipe):
    # (5.1) Define the class constructor
    def __init__(self):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        A.recipe.__init__(self)

        self.the_props = A.cable_global_properties()
        self.the_props.set_property(Vm=-65, tempK=300, rL=35.4, cm=0.01)
        self.the_props.set_ion(
            ion="na", int_con=10, ext_con=140, rev_pot=50, method="nernst/na"
        )
        self.the_props.set_ion(ion="k", int_con=54.4, ext_con=2.5, rev_pot=-77)
        self.the_props.set_ion(ion="ca", int_con=5e-5, ext_con=2, rev_pot=132.5)
        self.the_props.catalogue.extend(A.allen_catalogue(), "")

    # (5.2) Override the num_cells method
    def num_cells(self):
        return 1

    # (5.3) Override the cell_kind method
    def cell_kind(self, _):
        return A.cell_kind.cable

    # (5.4) Override the cell_description method
    def cell_description(self, _):
        return cell

    # (5.5) Override the probes method
    def probes(self, _):
        return [A.cable_probe_membrane_voltage('"custom_terminal"', "Um")]

    # (5.6) Override the global_properties method
    def global_properties(self, gid):
        return self.the_props


# Instantiate recipe
recipe = single_recipe()

# (6) Create a simulation
sim = A.simulation(recipe)

# Instruct the simulation to record the spikes and sample the probe
sim.record(A.spike_recording.all)

handle = sim.sample((0, "Um"), A.regular_schedule(0.02 * U.ms))

# (7) Run the simulation
sim.run(tfinal=100 * U.ms, dt=0.025 * U.ms)

# (8) Print or display the results
spikes = sim.spikes()
print(len(spikes), "spikes recorded:")
for (gid, lid), t in spikes:
    print(f" * t={t:.3f}ms gid={gid} lid={lid}")

data = []
meta = []
for d, m in sim.samples(handle):
    data.append(d)
    meta.append(m)

df_list = []
for i in range(len(data)):
    df_list.append(
        pd.DataFrame(
            {
                "t/ms": data[i][:, 0],
                "U/mV": data[i][:, 1],
                "Location": str(meta[i]),
                "Variable": "voltage",
            }
        )
    )
df = pd.concat(df_list, ignore_index=True)
sns.relplot(
    data=df,
    kind="line",
    x="t/ms",
    y="U/mV",
    hue="Location",
    col="Variable",
    errorbar=None,
).savefig("single_cell_recipe_result.svg")
