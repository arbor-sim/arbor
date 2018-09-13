#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np

import pycontra
import pynesci

receiver = pycontra.ZMQTransportRelay(pycontra.ZMQTransportType.Client,
                                      "tcp://localhost:5555", True)

master_node = pycontra.Node()
multimeter = pynesci.consumer.ArborMultimeter('some_name')
multimeter.SetNode(master_node)


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

ax1.set_ylim([-200, 200])
ax1.set_title("Ring example")
ax1.set_xlabel("Time")
ax1.set_ylabel("voltage")

counter = 0

while True:
    nodes = receiver.Receive()

    for node in nodes:
        master_node.Update(node)

        if counter > 100:
            timesteps = multimeter.GetTimesteps()
            attribute = 'voltage'
            neuron_ids = []
            if len(timesteps) > 0:
                neuron_ids = multimeter.GetNeuronIds(timesteps[0], attribute)

            ax1.clear()

            for neuron_id in neuron_ids:
                values = multimeter.GetTimeSeriesData(attribute, neuron_id)
                values = values[-100]
                times = [float(t) for t in timesteps]
                times = times[-100]
                ax1.plot(times, values)
            
            plt.show(block=False)
            fig.canvas.draw()

        counter = counter + 1
        print(counter)

    