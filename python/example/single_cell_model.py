#!/usr/bin/env python3

import arbor
import pandas, seaborn # You may have to pip install these.

# (1) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm
tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

# (2) Define the soma and its center
labels = arbor.label_dict({'soma':   '(tag 1)',
                           'center': '(location 0 0.5)'})

# (3) Create cell and set properties
cell = arbor.cable_cell(tree, labels)
cell.set_properties(Vm=-40)
cell.paint('"soma"', 'hh')
cell.place('"center"', arbor.iclamp( 10, 2, 0.8))
cell.place('"center"', arbor.spike_detector(-10))

# (4) Make single cell model.
m = arbor.single_cell_model(cell)

# (5) Attach voltage probe sampling at 10 kHz (every 0.1 ms).
m.probe('voltage', '"center"', frequency=10000)

# (6) Run simulation for 30 ms of simulated activity.
m.run(tfinal=30)

# (7) Print spike times, if any.
if len(m.spikes)>0:
    print('{} spikes:'.format(len(m.spikes)))
    for s in m.spikes:
        print('{:3.3f}'.format(s))
else:
    print('no spikes')

# (8) Plot the recorded voltages over time.
print("Plotting results ...")
seaborn.set_theme() # Apply some styling to the plot
df = pandas.DataFrame({'t/ms': m.traces[0].time, 'U/mV': m.traces[0].value})
seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV").savefig('single_cell_model_result.svg')

# (9) Optionally, you can store your results for later processing.
df.to_csv('single_cell_model_result.csv', float_format='%g')
