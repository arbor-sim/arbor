import matplotlib.pyplot as plt

# reads one column from file
def read_file(file_name):
    result = []

    file = open(file_name, 'r')

    for line in file:
        result += [float(line.split()[0])]
    
    file.close()


    return result

# reads two columns from file
def read_voltage_file(file_name):

    times = []
    voltages = []

    file = open(file_name, 'r')
    
    for line in file:
        times += [float(line.split()[0])]
        voltages += [float(line.split()[1])]

    file.close()

    return (times, voltages)

# read input spikes, out spikes and voltage from files
in_spikes = read_file("lif_neuron_input_spikes.txt")
out_spikes = read_file("lif_neuron_output_spikes.txt")
(times, voltages) = read_voltage_file("lif_neuron_voltage.txt")

fig = plt.figure(figsize=(7,7))

fig.suptitle(r'LIF neuron', fontsize=20)

# plot out spikes (spikes produced by LIF neuron)
ax_out = fig.add_subplot(3, 1, 2)

ax_out.set_xlabel("Output spikes time [ms]", fontsize=15)

for time in out_spikes:
    ax_out.axvline(time)

# plot in spikes (spikes that neuron LIF has received, i.e. events)
ax_in = fig.add_subplot(3, 1, 3, sharex = ax_out)

ax_in.set_xlabel("Input spikes time [ms]", fontsize=15)

for time in in_spikes:
    ax_in.axvline(time)

# plot voltage of LIF neuron
ax_voltage = fig.add_subplot(3, 1, 1, sharex = ax_out)
ax_voltage.set_xlabel("Membrane potential [mV]", fontsize=15)

ax_voltage.plot(times, voltages)

fig.subplots_adjust(hspace=.5)

plt.rcParams.update({'font.size': 22})


fig.savefig("lif_neuron_spikes.pdf")
