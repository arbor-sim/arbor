# 'single' example.

Example of simulating a single neuron with morphology described by an SWC file.

A cell is constructed from a supplied morphology with H–H channels
on the soma and passive channels on the dendrites. A simple exponential
synapse is added at the end of the last dendrite in the morphology,
and is triggered at time t = 1 ms.

The simulation outputs a trace of the soma membrane voltage in a simple CSV
format.

## Features

The example demonstrates the use of:

* Generating a morphology from an SWC file.
* Using a morphology to construct a cable cell.
* Injecting an artificial spike event into the simulation.
* Adding a voltage probe to a cell and running a sampler on the simulation.

## Running the example

By default, `single-cell` will simulate a 'ball-and-stick' neuron for 20 ms,
with a maxium dt of 0.025 ms and samples taken every 0.1 ms. The default
synaptic weight is 0.01 µS.

### Command line options

| Option                | Effect |
|-----------------------|--------|
| -m, --morphology FILE | Load the morphology from FILE in SWC format |
| -d, --dt TIME         | Set the maximum integration time step [ms] |
| -t, --t-end TIME      | Set the simulation duration [ms] |
| -w, --weight WEIGHT   | Set the synaptic weight [µS] |

