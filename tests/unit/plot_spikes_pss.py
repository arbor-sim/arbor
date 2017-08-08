import matplotlib.pyplot as plt

plt.style.use('ggplot')

# reads one column from file
def read_file(file_name):
    result = []
    file = open(file_name, 'r')

    for line in file:
        result += [float(line.split()[0])]

    file.close()
    return result

spikes = read_file("pps_spikes.txt")

fig = plt.figure()

fig.suptitle(r'Poisson neuron with rate 5Hz', fontsize=20)

# plot out spikes (spikes produced by Poisson(rate = 5) neuron)
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel("Time [s]", fontsize=13)
ax.set_ylabel("Number of spikes", fontsize=13)

# horizontal line showing the expected number of spikes
x = range(-3, 503)
ax.plot(x, [250 for _ in x], label=r'$\mathbb{E}\left[ \# spikes\right]$')

# histogram of produced spikes
ax.hist(spikes, bins=10)

ax.legend(fontsize=13)

plt.rcParams.update({'font.size': 22})

fig.savefig("pps_spikes.pdf")
