import matplotlib.pyplot as plt
from matplotlib import gridspec
import random

# reads two columns from file
def read_file(file_name):
    
    n_cores = []
    setup_time = []
    init_time = []
    sim_time = []
    
    file = open(file_name, 'r')
    
    for line in file:
        n_cores += [float(line.split()[0])]
        setup_time += [float(line.split()[1])]
        init_time += [float(line.split()[2])]
        sim_time += [float(line.split()[3])]

    file.close()

    return (n_cores, setup_time, init_time, sim_time)

# PLOTTING INITIALIZATION AND SIMULATION TIME
n_exc = [100, 1000]

fig = plt.figure(figsize=(8, 8))
#ax_setup = fig.add_subplot(311)
ax_init = fig.add_subplot(221)
ax_sim = fig.add_subplot(223, sharex=ax_init)

ax_init_speedup = fig.add_subplot(222)
ax_sim_speedup = fig.add_subplot(224, sharex=ax_init_speedup)

for n in n_exc:
    lab = "#neurons = " + str(n)
    n_cores, setup_time, init_time, sim_time = read_file("./scaling_cores_" + str(n) + ".txt")
    ax_init.plot(n_cores, init_time, label=lab)
    ax_sim.plot(n_cores, sim_time, label=lab)
    
    init_speedup = [init_time[0] / x for x in init_time]
    sim_speedup = [sim_time[0] / x for x in sim_time]
    
    ax_init_speedup.plot(n_cores, init_time, label=lab)
    ax_sim_speedup.plot(n_cores, sim_time, label=lab)

ax_init_speedup.set_xscale('log')
ax_sim_speedup.set_xscale('log')
ax_init_speedup.set_yscale('log')
ax_sim_speedup.set_yscale('log')

ax_init.legend()
ax_sim.legend()
ax_init_speedup.legend()
ax_sim_speedup.legend()

fsize = 15

ax_init.set_xlabel("Number of cores", fontsize=fsize)
ax_init.set_ylabel("Initialization time [s]", fontsize=fsize)
ax_sim.set_xlabel("Number of cores", fontsize=fsize)
ax_sim.set_ylabel("Simulation time [s]", fontsize=fsize)
ax_sim_speedup.set_xlabel("Log(Number of cores)", fontsize=fsize)
ax_sim_speedup.set_ylabel("Log(Sim time)", fontsize=fsize)
ax_init_speedup.set_xlabel("Log(Number of cores)", fontsize=fsize)
ax_init_speedup.set_ylabel("Log(Init time)", fontsize=fsize)

fig.suptitle("Varying #cores for #ranks=1", fontsize=22)

fig.subplots_adjust(hspace=.3)
fig.subplots_adjust(wspace=.3)
#fig.tight_layout()

fig.savefig("./varying_cores.pdf")

