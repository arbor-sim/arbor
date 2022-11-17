#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.

import arbor
import random
import multiprocessing
import numpy # You may have to pip install these.
#import pandas  # You may have to pip install these.
#import seaborn  # You may have to pip install these.
import matplotlib.pyplot as plt

# (x) Set simulation paramters

# total simulation time (ms)
tfinal = 10000 #59000
# spike response delay (ms)
D = 13.7
# spike frequency in Hertz
f = 1.0
# time lag resolution
stdp_dt_step = 5.0
# maximum time lag
stdp_max_dt = 10.0
# ensemble size per initial value
ensemble_per_rho_0 = 100 #100

# time lags between spike pairs (post-pre: < 0, pre-post: > 0)
stdp_dt = numpy.arange(-stdp_max_dt, stdp_max_dt+stdp_dt_step, stdp_dt_step)
# list of initial values for 2 states
rho_0 = ([0]*ensemble_per_rho_0 + [1]*ensemble_per_rho_0)
# we need a synapse for each sample path 
num_synapses = len(rho_0)
print(rho_0)

# (x) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm
tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

# (x) Define the soma and its midpoint
labels = arbor.label_dict({"soma": "(tag 1)", "midpoint": "(location 0 0.5)"})

# (x) Create extended catalogue including stochastic mechanisms
cable_properties = arbor.neuron_cable_properties()
cable_properties.catalogue = arbor.default_catalogue()
cable_properties.catalogue.extend(arbor.stochastic_catalogue(), "")

# (x) Create and set up a decor object
decor = (
    arbor.decor()
    .set_property(Vm=-40)
    .paint('"soma"', arbor.density("pas"))
    .place('"midpoint"', arbor.synapse('expsyn'), "driving_synapse")
    .place('"midpoint"', arbor.threshold_detector(-10), "detector")
)
for i in range(num_synapses):
    mech = arbor.mechanism('calcium_based_synapse')
    mech.set("rho_0", rho_0[i])
    decor.place('"midpoint"', arbor.synapse(mech), f"calcium_synapse_{i}")

# (x) Create cell and the single cell model based on it
cell = arbor.cable_cell(tree, decor, labels)

# (x) Create strong enough stimulus
stimulus_times = numpy.arange(abs(min(stdp_dt)), tfinal, f*1000)
stimulus_generator = arbor.event_generator("driving_synapse", 1., arbor.explicit_schedule(stimulus_times))
#generators = []
#generators.append(arbor.event_generator("driving_synapse", 1., arbor.explicit_schedule(stimulus_times)))
#for i in range(num_synapses):
#    # zero weight -> just modify synaptic weight via stdp
#    stdp = arbor.event_generator(f"calcium_synapse_{i}", 0., arbor.explicit_schedule([st - self.dt + D for st in stimulus_times]))
##            generators.append(stdp)

# (x) Recipe class
class stpd_recipe(arbor.recipe):
    def __init__(self, cell, props, gens):
        arbor.recipe.__init__(self)
        self.the_cell = cell
        self.the_props = props
        self.the_gens = gens

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        return self.the_cell

    def global_properties(self, kind):
        return self.the_props

    def probes(self, gid):
        return [
            arbor.cable_probe_point_state_cell("calcium_based_synapse", "rho")
        ]

    def event_generators(self, gid):
        return self.the_gens

def run(time_lag):

    print(time_lag)
    generators = [ stimulus_generator ]
    
    for i in range(num_synapses):
        # zero weight -> just modify synaptic weight via stdp
        generators.append(
            arbor.event_generator(
                f"calcium_synapse_{i}",
                0.,
                arbor.explicit_schedule([st - time_lag + D for st in stimulus_times])
            )
        )

    # create recipe
    recipe = stpd_recipe(cell, cable_properties, generators)

    # select one thread and no GPU
    alloc = arbor.proc_allocation(threads=1, gpu_id=None)
    context = arbor.context(alloc, mpi=None)
    domains = arbor.partition_load_balance(recipe, context)

    # get random seed
    random_seed = random.getrandbits(64)
    print("random_seed = " + str(random_seed))

    # create simulation
    sim = arbor.simulation(recipe, context, domains, seed=random_seed)
    
    handle = sim.sample((0, 0), arbor.regular_schedule(0.1))

    sim.run(tfinal, 0.1)

    data, meta = sim.samples(handle)[0]

    #print(data)


    # compute mean and standard deviation
    data_down = data[:,1:ensemble_per_rho_0+1]
    data_up = data[:,ensemble_per_rho_0+1:]

    mu_down = numpy.mean(data_down, axis=1)
    mu_up   = numpy.mean(data_up, axis=1)
    std_down = numpy.std(data_down, axis=1)
    std_up   = numpy.std(data_up, axis=1)
    #print(mu_down)
    #print(mu_up)
    #print()
    return numpy.stack((data[:,0], mu_up, mu_down, std_up, std_down, data_up[:,0],
                              data_down[:,0]), axis=1)

    ##print(d)
    ##print(data_down < 0.5)
    ##print(numpy.mean(data_down > 0.5, axis=1))
    ##print(numpy.mean(data_up < 0.5, axis=1))

    ##fmri = seaborn.load_dataset("fmri")
    ##print(fmri)
    ##df = pandas.DataFrame(data, colums=['ms', 
    ##print(['ms']+['s_' + str(i + 1) for i in range(data.shape[1]-1)])
    ##df = pandas.DataFrame(data, columns=['ms']+['s_' + str(i + 1) for i in range(data.shape[1]-1)])
    #df = pandas.DataFrame()
    #for i in range(data_down.shape[1]):
    #    df = pandas.concat([df, pandas.DataFrame({
    #        "ms": data[:,0],
    #        "rho": data_down[:,i],
    #        "rho_0":0,
    #        "s": f"d_{i}",
    #        "lag": time_lag})])
    #for i in range(data_up.shape[1]):
    #    df = pandas.concat([df, pandas.DataFrame({
    #        "ms": data[:,0],
    #        "rho": data_up[:,i],
    #        "rho_0": 1,
    #        "s": f"u_{i}",
    #        "lag": time_lag})])
    ##df2 = pandas.DataFrame({"ms": data[:,0], "rho": data[:,2], "rho_0": 0, "s": "s_1"})
    ##print(df)

    ##seaborn.set_theme()  # Apply some styling to the plot
    ##seaborn.relplot(data=df, kind="line", x="ms", y="rho", hue="rho_0", style="rho_0").savefig("calcium_stdp.svg")
    ##seaborn.relplot(data=df, kind="line", x="ms", y="rho",
    ##                col="time").savefig("calcium_stdp.svg")
    #return df

def plot(data):
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(20,10), constrained_layout=False)

    ax.plot(data[:,0], data[:,1])
    #ax.plot(data[:,0], data[:,5])
    ax.fill_between(data[:,0], (data[:,1]-data[:,3]), (data[:,1]+data[:,3]), color='b', alpha=.1)

    ax.plot(data[:,0], data[:,2])
    #ax.plot(data[:,0], data[:,6])
    ax.fill_between(data[:,0], (data[:,2]-data[:,4]), (data[:,2]+data[:,4]), color='r', alpha=.1)

    ax.set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig("calcium_stdp.png", bbox_inches="tight")

result = run(-10)
print(result)
plot(result)
#with multiprocessing.Pool() as p:
#    results = p.map(run, stdp_dt)



#print(results)
#df = results[0] #pandas.concat(results)
#seaborn.set_theme()  # Apply some styling to the plot
#plt = seaborn.lineplot(data=df, x="ms", y="rho",
#    hue="rho_0", 
#    style="lag",
#    errorbar=("sd"))
#fig = plt.get_figure()
#fig.savefig("calcium_stdp.svg")
