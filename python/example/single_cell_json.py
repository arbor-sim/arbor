import arbor
import pandas
import seaborn
import sys

if len(sys.argv) < 4:
    print("Pass default parameter file, decor file and SWC file to the program")
    sys.exit(0)

defaults_file = sys.argv[1]
decor_file    = sys.argv[2]
swc_file      = sys.argv[3]

# Load inputs from file
defaults = arbor.load_default_parameters(defaults_file)
decor    = arbor.load_decor(decor_file)
morph    = arbor.load_swc_arbor(swc_file)

# Test round trip of decor and default parameter descriptions
arbor.store_decor(decor, "decor_out.json")
arbor.store_default_parameters(defaults, "defaults_out.json")

# Define the regions and locsets in the model.
# These need to include the definitions of the region strings in the decor file
defs = {'soma': '(tag 1)',  # soma has tag 1 in swc files.
        'axon': '(tag 2)',  # axon has tag 2 in swc files.
        'dend': '(tag 3)',  # dendrites have tag 3 in swc files.
        'apic': '(tag 4)',  # apical dendrites have tag 4 in swc files.
        'all' : '(all)',    # all the cell
        'root': '(root)',   # the start of the soma in this morphology is at the root of the cell.
        'mid_soma': '(location 0 0.5)'
        } # end of the axon.
labels = arbor.label_dict(defs)

# Extend decor with discretization policy
decor.discretization(arbor.cv_policy_max_extent(0.5))

# Extend decor with spike detector and current clamp.
decor.place('"root"', arbor.spike_detector(-10))
decor.place('"mid_soma"', arbor.iclamp(0, 3, current=3.5))

# Combine morphology with region and locset definitions and decoration info to make a cable cell.
cell = arbor.cable_cell(morph, labels, decor)

# Make single cell model.
model = arbor.single_cell_model(cell)

# Set the model default parameters
model.properties.set_property(defaults)

# Extend the default catalogue
model.catalogue.extend(arbor.bbp_catalogue(), "")

# Attach voltage probes that sample at 50 kHz.
model.probe('voltage', where='"root"',  frequency=50000)
model.probe('voltage', where='"mid_soma"', frequency=50000)

# Simulate the cell for 20 ms.
tfinal=20
model.run(tfinal)

# Print spike times.
if len(model.spikes)>0:
    print('{} spikes:'.format(len(model.spikes)))
    for s in model.spikes:
        print('  {:7.4f}'.format(s))
else:
    print('no spikes')

# Plot the recorded voltages over time.
df = pandas.DataFrame()
for t in model.traces:
    df=df.append(pandas.DataFrame({'t/ms': t.time, 'U/mV': t.value, 'Location': str(t.location), 'Variable': t.variable}))

seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Location",col="Variable",ci=None).savefig('single_cell_json_result.svg')
