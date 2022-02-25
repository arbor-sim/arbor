#!/usr/bin/env python3

import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(spikes, tfinal, xmin, xmax):

    df = pd.read_csv(spikes, names=["neuron", "spike_time"], sep=' ')

    data = {nrn : df_.spike_time.to_numpy() for nrn, df_ in df.groupby('neuron')}

    print("average firing rate [Hz]:", np.mean(np.array(list(map(len, data.values()))) / (tfinal/1000)))

    random_choice = random.sample(list(data.values()), 50)

    fig, axes = plt.subplots(2)

    for i, spikes in enumerate(random_choice):
        axes[0].plot(spikes, [i]*len(spikes), marker='|', linestyle="None")

    df.spike_time.hist(bins=np.arange(0, tfinal, 0.1), ax=axes[1])

    for ax in axes:
        ax.set_xlim(xmin, xmax)

    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('spikes')
    parser.add_argument('tfinal', type=float)
    parser.add_argument('xmin', type=float)
    parser.add_argument('xmax', type=float)

    args = parser.parse_args()

    plot(args.spikes, args.tfinal, args.xmin, args.xmax)
