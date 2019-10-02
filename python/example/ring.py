import sys
import arbor
import matplotlib.pyplot as plt

class ring_recipe (arbor.recipe):

    def __init__(self, n=4):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)
        self.ncells = n
        self.params = arbor.cell_parameters()

    # The num_cells method that returns the total number of cells in the model
    # must be implemented.
    def num_cells(self):
        return self.ncells

    # The cell_description method returns a cell
    def cell_description(self, gid):
        return arbor.make_cable_cell(gid, self.params)

    def num_targets(self, gid):
        return 1

    def num_sources(self, gid):
        return 1

    # The kind method returns the type of cell with gid.
    # Note: this must agree with the type returned by cell_description.
    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    # Make a ring network
    def connections_on(self, gid):
        src = (gid-1)%self.ncells
        w = 0.01
        d = 10
        return [arbor.connection(arbor.cell_member(src,0), arbor.cell_member(gid,0), w, d)]

    # Attach a generator to the first cell in the ring.
    def event_generators(self, gid):
        if gid==0:
            sched = arbor.explicit_schedule([1])
            return [arbor.event_generator(arbor.cell_member(0,0), 0.1, sched)]
        return []

    # Define one probe (for measuring voltage at the soma) on each cell.
    def num_probes(self, gid):
        return 1

    def get_probe(self, id):
        loc = arbor.location(0, 0) # at the soma
        return arbor.cable_probe('voltage', id, loc)

context = arbor.context(threads=4, gpu_id=None)
print(context)

meters = arbor.meter_manager()
meters.start(context)

recipe = ring_recipe(10)
print(f'{recipe}')

meters.checkpoint('recipe-create', context)

decomp = arbor.partition_load_balance(recipe, context)
print(f'{decomp}')

hint = arbor.partition_hint()
hint.prefer_gpu = True
hint.gpu_group_size = 1000
print(f'{hint}')

hints = dict([(arbor.cell_kind.cable, hint)])
decomp = arbor.partition_load_balance(recipe, context, hints)
print(f'{decomp}')

meters.checkpoint('load-balance', context)

sim = arbor.simulation(recipe, decomp, context)

meters.checkpoint('simulation-init', context)

spike_recorder = arbor.attach_spike_recorder(sim)

pid = arbor.cell_member(0,0) # cell 0, probe 0
# Attach a sampler to the voltage probe on cell 0.
# Sample rate of 1 sample every ms.
sampler = arbor.attach_sampler(sim, 1, pid)

sim.run(100)
print(f'{sim} finished')

meters.checkpoint('simulation-run', context)

print(f'{arbor.meter_report(meters, context)}')

for sp in spike_recorder.spikes:
    print(sp)

print('voltage samples for probe id ', end = '')
print(pid, end = '')
print(':')

time = []
value = []
for sa in sampler.samples(pid):
    print(sa)
    time.append(sa.time)
    value.append(sa.value)

# plot the recorded voltages over time
fig, ax = plt.subplots()
ax.plot(time, value)
ax.set(xlabel='time (ms)', ylabel='voltage (mV)', title='ring demo')
ax.legend(['voltage'])
plt.xlim(0,100)
ax.grid()
fig.savefig("voltages.png", dpi=300)
