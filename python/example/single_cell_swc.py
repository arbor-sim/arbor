#!/usr/bin/env python3

# NOTE: deprecating spherical roots changes the behavior of this model.
# There is no soma, because only the root sample has tag 1, which will be
# ignored as it is always the proximal end of any cable segment.
# The fix is to:
#   - Write an swc interpreter that inserts a cylinder with the
#     appropriate properties.
#   - Extend the cable-only descriptions to handle detached cables, to
#     preserve surface area and correct starting locations of cables
#     attached to the soma.

import arbor as A
from arbor import units as U
import pandas as pd
import seaborn as sns
import sys

# Load a cell morphology from an swc file.
# Example present here: single_cell_detailed.swc
if len(sys.argv) < 2:
    print("No SWC file passed to the program")
    sys.exit(0)

filename = sys.argv[1]
morpho = A.load_swc_arbor(filename)

# Define the regions and locsets in the model.
labels = A.label_dict(
    {
        "root": "(root)",  # the start of the soma in this morphology is at the root of the cell.
        "stim_site": "(location 0 0.5)",  # site for the stimulus, in the middle of branch 0.
        "axon_end": '(restrict-to (terminal) (region "axon"))',
    }  # end of the axon.
).add_swc_tags()  # Finally, add the SWC default labels.

decor = (
    A.decor()
    # Set initial membrane potential to -55 mV
    .set_property(Vm=-55 * U.mV)
    # Use Nernst to calculate reversal potential for calcium.
    .set_ion("ca", method=A.mechanism("nernst/x=ca"))
    # hh mechanism on the soma and axon.
    .paint('"soma"', A.density("hh"))
    .paint('"axon"', A.density("hh"))
    # pas mechanism the dendrites.
    .paint('"dend"', A.density("pas"))
    # Increase resistivity on dendrites.
    .paint('"dend"', rL=500 * U.Ohm * U.cm)
    # Attach stimuli that inject 4 nA current for 1 ms, starting at 3 and 8 ms.
    .place('"root"', A.iclamp(10 * U.ms, 1 * U.ms, current=5 * U.nA), "iclamp0")
    .place('"stim_site"', A.iclamp(3 * U.ms, 1 * U.ms, current=0.5 * U.nA), "iclamp1")
    .place('"stim_site"', A.iclamp(10 * U.ms, 1 * U.ms, current=0.5 * U.nA), "iclamp2")
    .place('"stim_site"', A.iclamp(8 * U.ms, 1 * U.ms, current=4 * U.nA), "iclamp3")
    # Detect spikes at the soma with a voltage threshold of -10 mV.
    .place('"axon_end"', A.threshold_detector(-10 * U.mV), "detector")
    # Create the policy used to discretise the cell into CVs.
    # Use a single CV for the soma, and CVs of maximum length 1 μm elsewhere.
    .discretization('(replace (single (region "soma")) (max-extent 1.0))')
)

# Combine morphology with region and locset definitions to make a cable cell.
cell = A.cable_cell(morpho, decor, labels)

# Make single cell model.
m = A.single_cell_model(cell)

# Attach voltage probes that sample at 50 kHz.
m.probe("voltage", tag="Um-root", where='"root"', frequency=50 * U.kHz)
m.probe("voltage", tag="Um-stim", where='"stim_site"', frequency=50 * U.kHz)
m.probe("voltage", tag="Um-axon", where='"axon_end"', frequency=50 * U.kHz)

# Simulate the cell for 15 ms.
m.run(15 * U.ms)
print("Simulation done.")

# Print spike times.
if len(m.spikes) > 0:
    print("{} spikes:".format(len(m.spikes)))
    for s in m.spikes:
        print("  {:7.4f}".format(s))
else:
    print("no spikes")

# Plot the recorded voltages over time.
print("Plotting results ...")
df_list = []
for t in m.traces:
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
).savefig("single_cell_swc.svg")
