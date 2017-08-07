'''
   Makes the raster plot where each point (x, y) represents the spike of neuron y in time x.
   The points are colored differently, depending on which neuronal population they belong to.
   Expects the number of excitatory (-n), inhibitory (-m) and external Poisson (-e) neurons as command line arguments.
'''


import matplotlib.pyplot as plt
from matplotlib import gridspec
import random
from argparse import ArgumentParser
import sys

# Three parameters are expected as command line arguments.
# These arguments must match the arguments of the simulation of Brunel network
# that was previously run.
print("Expecting the number of excitatory (-n), inhibitory (-m) and external Poisson (-e) neurons as command line arguments.")

parser = ArgumentParser()
parser.add_argument("-n", "--n_exc", type=int, default=400, help="Number of excitatory neurons.")
parser.add_argument("-m", "--n_inh", type=int, default=100, help="Number of inhibitory neurons.")
parser.add_argument("-e", "--n_ext", type=int, default=400, help="Number of Poisson (external) neurons.")

args = parser.parse_args()

# reads two columns from file
def read_file(file_name):
    
    ids = []
    times = []
    
    file = open(file_name, 'r')
    
    for line in file:
        ids += [float(line.split()[0])]
        times += [float(line.split()[1])]
    
    file.close()

    return (ids, times)

nexc = args.n_exc
ninh = args.n_inh
next = args.n_ext

(ids, times) = read_file("../../build/miniapp/brunel/spikes_0.gdf")

fig = plt.figure(figsize=(7, 8))
fig.suptitle("Brunel Network Spikes", fontsize=22)
ax = fig.add_subplot(1, 1, 1)


times_exc, ids_exc, times_inh, ids_inh, times_ext, ids_ext = [], [], [], [], [], []

for (id, time) in zip(ids, times):
    if id < nexc: # excitatory
        ids_exc += [id]
        times_exc += [time]

    elif id < nexc + ninh: # inhibitory
        ids_inh += [id]
        times_inh += [time]
    else: # Poisson (external)
        # Plot only a half of Poisson spikes for readability reasons.
        if random.random() < 0.5:
            ids_ext += [id]
            times_ext += [time]

ax.scatter(times_exc, ids_exc, s=0.2, label="Excitatory")
ax.scatter(times_inh, ids_inh, s=0.2, label="Inhibitory")
ax.scatter(times_ext, ids_ext, s=0.2, label="Poisson")

lgnd = ax.legend(loc="upper right")

#change the marker size manually for both lines
lgnd.legendHandles[0]._sizes = [50]
lgnd.legendHandles[1]._sizes = [50]
lgnd.legendHandles[2]._sizes = [50]

ax.set_xlabel("Spike Time [ms]", fontsize=15)
ax.set_ylabel("Neuron ID", fontsize=15)


fig.savefig("brunel_spikes.pdf")
