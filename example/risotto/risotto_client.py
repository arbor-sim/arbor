import pycontra
import pynesci

receiver = pycontra.ZMQTransportRelay(pycontra.ZMQTransportType.Client,
                                      "tcp://localhost:5555", True)

multimeter = pynesci.consumer.ArborMultimeter('some_name')

while True:
    nodes = receiver.Receive()

    for node in nodes:
        multimeter.SetNode(node)
        for timestep in multimeter.GetTimesteps():
            print(timestep)
            for neuron_id in multimeter.GetNeuronIds(timestep, 'voltage'):
                voltage = multimeter.GetDatum(timestep, 'voltage', neuron_id)
                print(neuron_id + ': ' + str(voltage))
            print('')

    