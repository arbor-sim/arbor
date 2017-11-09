# Arbor miniapp

The miniapp is a simple benchmark for the Arbor library.

## the model

The model is very simple, with three parameters that describe the network, and two for controlling time stepping.

### network/model parameters

The following parameters are used to describe the size, connectivity and resolution of the model:

- `cells` is the total number of cells in the network.
- `synapses_per_cell` the number of synapses per cell, must be in the range `[0,cells-1]`
- `compartments` the number of compartments per segment.
- `all_to_all` toggle whether to make an all to all network

All cells have identical morphology, a soma with a dendrite attached. The dendrite branches as illustrated (roughly) below


```
s = soma
. = branching point of dendrite

         /
        /
s------.
        \
         \
```

The disrcetization of each cell is controlled with the __compartments__ parameter.
For example, when `compartments=100`, the total number of compartments in each cell is 301: 1 for the soma, and 100 for each of the dendrite segments.

The `synapses_per_cell` parameter is in the range `[0,cells-1]`.
If it is zero, then there are no connections between the cells (not much of a network).
By default, the source gid of each synapse is chosen at random from the global set of cells (excluding the cell of the synapse).
If the `all_to_all` flag is set, `synapses_per_cell` is set to `cells-1`, i.e. one connection for each cell (excluding the cell of the synapse)

Note that the to avoid numerical instability, the number of synapses per cell should be greater than 200 (or zero!).
The number of synapses per cell required for stability is dependent on the number of compartments per segment (fewer compartments is more stable) and the time step size (smaller time step sizes increase stability).
If there are numeric instabilities the simulation will print a warning
```
warning: solution out of bounds
```

### time stepping parameters

The time stepping can be controlled via two parameters

- `dt` the time step size in ms (default `dt=0.025`)
- `tfinal` the length of the simulation in ms (default `tfinal=200`)

## configuration

There are two ways to specify the model properties, i.e. the number of cells, connections per cell, etc.

### command line arguments

- `-n integer` : `ncells`
- `-s integer` : `synapses_per_cell`
- `-c integer` : `compartments`
- `-d float`   : `dt`
- `-t float`   : `tfinal`
- `-i filename` : name of json file with parameters

For example

```
> ./miniapp.exe -n 1000 -s 500 -c 50 -t 100 -d 0.02
```

will run a simulation with 1000 cells, with 500 synapses per cell, 50 compartments per segment for a total of 100 ms with 0.02 ms time steps.

### input parameter file

To run the same simulation that we ran with the command line arguments with an input file:

```
> cat input.json
{
    "cells": 1000,
    "synapses": 500,
    "compartments": 50,
    "dt": 0.02,
    "tfinal": 100.0,
    "all_to_all": false
}
> ./miniapp.exe -i input.json
```
