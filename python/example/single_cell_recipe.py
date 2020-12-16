import arbor
import numpy, pandas, seaborn # You may have to pip install these.

# The corresponding generic recipe version of `single_cell_model.py`.

# (1) Define a recipe for a single cell and set of probes upon it.

class single_recipe (arbor.recipe):
    def __init__(self, cell, probes):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)
        self.the_cell = cell
        self.the_probes = probes
        self.the_props = arbor.neuron_cable_propetries()
        self.the_cat = arbor.default_catalogue()
        self.the_props.register(self.the_cat)

    def num_cells(self):
        return 1

    def num_sources(self, gid):
        return 1

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        return self.the_cell

    def probes(self, gid):
        return self.the_probes

    def global_properties(self, kind):
        return self.the_props

# (2) Create a cell.

# Morphology
tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

# Label dictionary
labels = arbor.label_dict()
labels['centre'] = '(location 0 0.5)'

# Decorations
decor = arbor.decor()
decor.set_property(Vm=-40)
decor.paint('(all)', 'hh')
decor.place('"centre"', arbor.iclamp( 10, 2, 0.8))
decor.place('"centre"', arbor.spike_detector(-10))

cell = arbor.cable_cell(tree, labels, decor)

# (3) Instantiate recipe with a voltage probe.

recipe = single_recipe(cell, [arbor.cable_probe_membrane_voltage('"centre"')])

# (4) Instantiate simulation and set up sampling on probe id (0, 0).

context = arbor.context()
domains = arbor.partition_load_balance(recipe, context)
sim = arbor.simulation(recipe, domains, context)

sim.record(arbor.spike_recording.all)
handle = sim.sample((0, 0), arbor.regular_schedule(0.1))

# (6) Run simulation for 30 ms of simulated activity and collect results.

sim.run(tfinal=30)
spikes = sim.spikes()
data, meta = sim.samples(handle)[0]

# (7) Print spike times, if any.

if len(spikes)>0:
    print('{} spikes:'.format(len(spikes)))
    for t in spikes['time']:
        print('{:3.3f}'.format(t))
else:
    print('no spikes')

# (8) Plot the recorded voltages over time.

print("Plotting results ...")
seaborn.set_theme() # Apply some styling to the plot
df = pandas.DataFrame({'t/ms': data[:, 0], 'U/mV': data[:, 1]})
seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV", ci=None).savefig('single_cell_recipe_result.svg')

# (9) Optionally, you can store your results for later processing.

df.to_csv('single_cell_recipe_result.csv', float_format='%g')
