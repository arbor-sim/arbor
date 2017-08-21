import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import random
import math

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
n_exc = [100, 1000, 10000]

fig = plt.figure(figsize=(8, 4))

ax_sim = fig.add_subplot(121)
ax_sim_log = fig.add_subplot(122)

all_times = []

for n in n_exc:
    lab = "#neurons = " + str(n)
    n_cores, setup_time, init_time, sim_time = read_file("./scaling_cores_" + str(n) + ".txt")
    ax_sim.plot(n_cores, sim_time, label=lab, marker='o', markersize=5)
    ax_sim_log.plot(n_cores, sim_time, label=lab, marker='o', markersize=5)
    all_times += sim_time

#print(all_times)

ax_sim.set_xticks(n_cores)
ax_sim.set_yticks(sim_time + [all_times[0]])
#ax_sim.set_yticklabels(all_times)

ax_sim_log.set_xscale('log')
ax_sim_log.set_yscale('log')
ax_sim_log.set_xticks(n_cores)
yticks = [x for (i, x) in enumerate(all_times) if i % 2 == 0]
yticks = [x for (i, x) in enumerate(yticks) if i != 2]
ax_sim_log.set_yticks(yticks)
ax_sim_log.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_sim_log.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax_sim_log.set_yticklabels(yticks)


ax_sim.legend()
ax_sim_log.legend()

fsize = 20

ax_sim.set_xlabel("Number of cores", fontsize=fsize)
ax_sim.set_ylabel("Simulation time [s]", fontsize=fsize)
ax_sim_log.set_xlabel("Log(Number of cores)", fontsize=fsize)
ax_sim_log.set_ylabel("Log(Sim time)", fontsize=fsize)

fig.suptitle("Varying #cores for #ranks=1", fontsize=22)

ax_sim.grid(True, linestyle='--',  linewidth=0.8)
ax_sim_log.grid(True, linestyle='--', linewidth=0.8)

fig.subplots_adjust(hspace=.3)
fig.subplots_adjust(wspace=.3)
fig.tight_layout(rect=[0, 0.03, 1, 0.93])

fig.savefig("./varying_cores.pdf")

