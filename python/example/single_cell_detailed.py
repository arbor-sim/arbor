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
).add_swc_tags()

# (3) Create and populate the decor.
decor = (
    A.decor()
    # Set the default properties of the cell (this overrides the model defaults).
    .set_property(Vm=-55 * U.mV)
    .set_ion(
        "na",
        int_con=10 * U.mM,
        ext_con=140 * U.mM,
        rev_pot=50 * U.mV,
        method="nernst/na",
    )
    .set_ion("k", int_con=54.4 * U.mM, ext_con=2.5 * U.mM, rev_pot=-77 * U.mV)
    # Override the cell defaults.
    .paint('"custom"', tempK=270 * U.Kelvin)
    .paint('"soma"', Vm=-50 * U.mV)
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

# (5) Construct the model
model = A.single_cell_model(cell)

# (6) Set the model default properties
model.properties.set_property(
    Vm=-65 * U.mV, tempK=300 * U.Kelvin, rL=35.4 * U.Ohm * U.cm, cm=0.01 * U.F / U.m2
)
model.properties.set_ion(
    "na", int_con=10 * U.mM, ext_con=140 * U.mM, rev_pot=50 * U.mV, method="nernst/na"
)
model.properties.set_ion(
    "k", int_con=54.4 * U.mM, ext_con=2.5 * U.mM, rev_pot=-77 * U.mV
)

# Extend the default catalogue with the Allen catalogue.
# The function takes a second string parameter that can prefix
# the name of the mechanisms to avoid collisions between catalogues
# in this case we have no collisions so we use an empty prefix string.
model.properties.catalogue.extend(A.allen_catalogue(), "")

# (7) Add probes.
# Add voltage probes on the "custom_terminal" locset
# which sample the voltage at 50 kHz
model.probe("voltage", where='"custom_terminal"', tag="Um", frequency=50 * U.kHz)

# (8) Run the simulation for 100 ms, with a dt of 0.025 ms
model.run(tfinal=100 * U.ms, dt=25 * U.us)

# (9) Print the spikes.
print(len(model.spikes), "spikes recorded:")
for s in model.spikes:
    print(f" * t={s:.3f} ms")

# (10) Plot the voltages
df_list = []
for t in model.traces:
    df_list.append(
        pd.DataFrame(
            {
                "t/ms": t.time,
                "U/mV": t.value,
                "Location": str(t.location),
                "Variable": t.variable,
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
).savefig("single_cell_detailed_result.svg")
