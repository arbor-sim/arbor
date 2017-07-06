import matplotlib.pyplot as plt
from matplotlib import gridspec
import random

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

nexc = 500
ninh = 100
next = 500

(ids, times) = read_file("brunel_network_spikes.txt")

fig = plt.figure(figsize=(7, 8))
fig.suptitle("Brunel Network Spikes", fontsize=22)
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax = fig.add_subplot(gs[0])


times_exc, ids_exc, times_inh, ids_inh, times_ext, ids_ext = [], [], [], [], [], []

for (id, time) in zip(ids, times):
    if id < nexc:
        ids_exc += [id]
        times_exc += [time]

    elif id < nexc + ninh:
        ids_inh += [id]
        times_inh += [time]
    else:
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

#ax.axhline(y=nex, c='k')
#ax.axhline(y=nex+ninh, c='k')
(time, voltage) = read_file("brunel_voltage.txt")

ax_voltage = fig.add_subplot(gs[1])
ax_voltage.plot(time, voltage)
ax_voltage.set_xlabel("Time [ms]", fontsize=15)
ax_voltage.set_ylabel("Voltage [mV]", fontsize=15)

fig.savefig("brunel_spikes.pdf")
