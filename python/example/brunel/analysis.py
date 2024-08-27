import numpy as np
import matplotlib.pyplot as plt
from parameters import *


def plot_raster_and_PSTH(times, sources, bin=30):
    plt.figure(figsize=(60, 80))

    plt.subplot(311)
    plt.plot(times, sources, "|", color="k")
    plt.ylim([100, 200])
    plt.xlim(0, tfinal)
    plt.ylabel("neuron ID")

    plt.subplot(312)
    plt.plot(times, sources, "|", color="k")
    plt.xlim(0, tfinal)
    plt.ylabel("neuron ID")

    plt.subplot(313)
    counts = np.histogram(times, range(0, tfinal))[0]
    lefts = np.histogram(times, range(0, tfinal))[1][0:-1]
    plt.bar(lefts, counts, color="k")
    plt.xlabel("time | ms")
    plt.ylabel("counts")
    plt.xlim(0, tfinal)

    plt.suptitle("raster plot and PSTH")
    plt.savefig("dynamics.svg")


def analysis_rate(sources, times, tfinal):
    rates = []

    for neuron in np.arange(0, NE + NI, 1):
        idx = np.where(sources == neuron + 1)
        time = times[idx]
        rate = len(time) / (tfinal / 1000.0)
        rates.append(rate)
    return rates


def plot_rate_histogram(sources, times, tfinal):
    rates = analysis_rate(sources, times, tfinal)
    plt.figure(2)
    plt.hist(rates)
    plt.xlabel("firing rate | Hz")
    plt.ylabel("counts")
    plt.savefig("rates.svg")


times = np.load("times.npy")
sources = np.load("sources.npy")
plot_raster_and_PSTH(times, sources)
plot_rate_histogram(sources, times, tfinal)
plt.show()
