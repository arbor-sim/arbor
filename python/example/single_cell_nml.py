#!/usr/bin/env python3
import arbor
from arbor import mechanism as mech
import pandas
import seaborn
import sys

# Load a cell morphology from an nml file.
# Example present here: morph.nml
if len(sys.argv) < 2:
    print("No NeuroML file passed to the program")
    sys.exit(0)

filename = sys.argv[1]

# Read the NeuroML morphology from the file.
morpho_nml = arbor.neuroml(filename)

# Read the morphology data associated with morphology "m1".
morpho_data = morpho_nml.morphology("m1")

# Get the morphology.
morpho = morpho_data.morphology

# Get the region label dictionaries associated with the morphology.
morpho_segments = morpho_data.segments()
morpho_named = morpho_data.named_segments()
morpho_groups = morpho_data.groups()

# Create new label dict with some locsets.
labels = arbor.label_dict(
    {
        "stim_site": "(location 1 0.5)",  # site for the stimulus, in the middle of branch 1.
        "axon_end": '(restrict (terminal) (region "axon"))',  # end of the axon.
        "root": "(root)",  # the start of the soma in this morphology is at the root of the cell.
    }
)
# Add to it all the NeuroML dictionaries.
labels.append(morpho_segments)
labels.append(morpho_named)
labels.append(morpho_groups)

# Optional: print out the regions and locsets available in the label dictionary.
print("Label dictionary regions: ", labels.regions, "\n")
print("Label dictionary locsets: ", labels.locsets, "\n")

decor = (
    arbor.decor()
    # Set initial membrane potential to -55 mV
    .set_property(Vm=-55)
    # Use Nernst to calculate reversal potential for calcium.
    .set_ion("ca", method=mech("nernst/x=ca"))
    # hh mechanism on the soma and axon.
    .paint('"soma"', arbor.density("hh"))
    .paint('"axon"', arbor.density("hh"))
    # pas mechanism the dendrites.
    .paint('"dend"', arbor.density("pas"))
    # Increase resistivity on dendrites.
    .paint('"dend"', rL=500)
    # Attach stimuli that inject 4 nA current for 1 ms, starting at 3 and 8 ms.
    .place('"root"', arbor.iclamp(10, 1, current=5), "iclamp0")
    .place('"stim_site"', arbor.iclamp(3, 1, current=0.5), "iclamp1")
    .place('"stim_site"', arbor.iclamp(10, 1, current=0.5), "iclamp2")
    .place('"stim_site"', arbor.iclamp(8, 1, current=4), "iclamp3")
    # Detect spikes at the soma with a voltage threshold of -10 mV.
    .place('"axon_end"', arbor.threshold_detector(-10), "detector")
    # Set discretisation: Soma as one CV, 1um everywhere else
    .discretization('(replace (single (region "soma")) (max-extent 1.0))')
)

# Combine morphology with region and locset definitions to make a cable cell.
cell = arbor.cable_cell(morpho, decor, labels)

print(cell.locations('"axon_end"'))

# Make single cell model.
m = arbor.single_cell_model(cell)

# Attach voltage probes that sample at 50 kHz.
m.probe("voltage", where='"root"', frequency=50)
m.probe("voltage", where='"stim_site"', frequency=50)
m.probe("voltage", where='"axon_end"', frequency=50)

# Simulate the cell for 15 ms.
tfinal = 15
m.run(tfinal)
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
        pandas.DataFrame(
            {
                "t/ms": t.time,
                "U/mV": t.value,
                "Location": str(t.location),
                "Variable": t.variable,
            }
        )
    )

df = pandas.concat(df_list, ignore_index=True)

seaborn.relplot(
    data=df,
    kind="line",
    x="t/ms",
    y="U/mV",
    hue="Location",
    col="Variable",
    errorbar=None,
).savefig("single_cell_nml.svg")
