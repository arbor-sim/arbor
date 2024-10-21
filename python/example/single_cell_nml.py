#!/usr/bin/env python3
import arbor as A
from arbor import units as U
import pandas as pd
import seaborn as sns
import sys
from pathlib import Path

# Load a cell morphology from an nml file.
# Example present here: morph.nml
if len(sys.argv) == 1:
    print("No NML file passed to the program, using default.")
    filename = Path(__file__).parent / "morph.nml"
elif len(sys.argv) == 2:
    filename = Path(sys.argv[1])
else:
    print("Usage: single_cell_nml.py [NML file name]")
    sys.exit(1)

# Read the NeuroML morphology from the file.
morpho_nml = A.neuroml(filename)

# Read the morphology data associated with morphology "m1".
morpho_data = morpho_nml.morphology("m1")

# Get the morphology.
morpho = morpho_data.morphology

# Create new label dict with some locsets and add to it all the NeuroML dictionaries.
labels = A.label_dict(
    {
        "stim_site": "(location 1 0.5)",  # site for the stimulus, in the middle of branch 1.
        "axon_end": '(restrict-to (terminal) (region "axon"))',  # end of the axon.
        "root": "(root)",  # the start of the soma in this morphology is at the root of the cell.
    }
)
labels.append(morpho_data.labels)

# Optional: print out the regions and locsets available in the label dictionary.
print("Label dictionary regions: ", labels.regions, "\n")
print("Label dictionary locsets: ", labels.locsets, "\n")

decor = (
    A.decor()
    # Set initial membrane potential to -55 mV
    .set_property(Vm=-55 * U.mV)
    # Use Nernst to calculate reversal potential for calcium.
    .set_ion("ca", method="nernst/x=ca")
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
)

# Set discretisation: Soma as one CV, 1um everywhere else
cvp = A.cv_policy('(replace (single (region "soma")) (max-extent 1.0))')

# Combine morphology with region and locset definitions to make a cable cell.
cell = A.cable_cell(morpho, decor, labels, cvp)

print(cell.locations('"axon_end"'))

# Make single cell model.
m = A.single_cell_model(cell)

# Attach voltage probes that sample at 50 kHz.
m.probe("voltage", where='"root"', tag="Um-root", frequency=50 * U.kHz)
m.probe("voltage", where='"stim_site"', tag="Um-stim", frequency=50 * U.kHz)
m.probe("voltage", where='"axon_end"', tag="Um-axon", frequency=50 * U.kHz)

# Simulate the cell for 15 ms.
m.run(15 * U.ms)
print("Simulation done.")

# Print spike times.
if len(m.spikes) > 0:
    print("{} spikes:".format(len(m.spikes)))
    for s in m.spikes:
        print(f"  {s:7.4f} ms")
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
).savefig("single_cell_nml.svg")
