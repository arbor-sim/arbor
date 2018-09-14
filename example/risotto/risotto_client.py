#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np

import pycontra
import pynesci

receiver = pycontra.ZMQTransportRelay(pycontra.ZMQTransportType.Client,
                                      "tcp://localhost:5555", True)


multimeter = pynesci.consumer.ArborMultimeter('some_name')

master_node = pycontra.Node()


fig = plt.figure(figsize=(15,7.5))
ax1 = fig.add_subplot(1,1,1)

ax1.set_ylim([-80, 50])
ax1.set_xlim([0, 10000])
ax1.set_title("Ring example")
ax1.set_xlabel("Time")
ax1.set_ylabel("voltage")


counter = 0
data_received = False
while True:
    nodes = receiver.Receive()

    for node in nodes:
        multimeter.SetNode(node)
        #master_node.Update(node)

        timesteps = multimeter.GetTimesteps()
        attribute = 'voltage'
        neuron_ids = []
        if len(timesteps) > 0:
            neuron_ids = multimeter.GetNeuronIds(timesteps[0], attribute)

        nr_lines = len(neuron_ids)
        for idx, neuron_id in enumerate(neuron_ids):

            values = multimeter.GetTimeSeriesData(attribute, neuron_id)
            times = [float(t) for t in timesteps]
            color = (idx * 1.0 /nr_lines, idx * 1.0 /nr_lines, idx * 1.0 /nr_lines )
            ax1.plot(times, values, color=color)

        plt.show(block=False)
        fig.canvas.draw()

        counter = counter + 1
        print(counter)

