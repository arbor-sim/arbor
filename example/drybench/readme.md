# Dryrun Example

A miniapp that demonstrates how to use dry-run mode to simulate the effect of communication scaling
with benchmark cells. This is useful for evaluating the communication overheads as a model
is weak-scaled over MPI.


It uses the `arb::tile` to build a network of `num_cells` cells per rank of type
`arb::benchmark_cell`. The network is translated over `ranks` domains using `arb::symmetric_recipe`.

The model of the *tile* can be configured using a json configuration file:

```
./drybench params.json
```

An example parameter file for a dry-run is:
```
{
    "name": "test",
    "num-cells": 100,
    "duration": 100,
    "min-delay": 10,
    "fan-in": 10,
    "realtime-ratio": 0.1,
    "spike-frequency": 20,
    "ranks": 10000
    "threads": 4,
}
```

The parameters in the file:
  * `name="default"`: a string with a name for the benchmark.
  * `num-cells=100`: the number of cells on a single tile.
    The total number of cells in the model = num-cells * ranks.
  * `duration=100`: the length of the simulated time interval, in ms.
  * `min-delay=10`: the minimum delay of the network.
  * `fan-in=5000`: the number of incoming connections on each cell.
  * `spike-frequency=20`: frequency (Hz) of the independent Poisson processes that
    generate spikes for each cell.
  * `realtime-ratio=0.1`: the ratio between time taken to advance a single cell in
    the simulation and the simulated time. For example, a value of 1 indicates
    that the cell is simulated in real time, while a value of 0.1 indicates
    that 10s can be simulated in a single second.
  * `ranks=1`: the number of domains to simulate.
  * `threads=available-threads-on-system`: the number of threads per rank: default is automatically detected.

The network is randomly connected with no self-connections and `fan-in`
incoming connections on each cell, with every connection having delay of `min-delay`,
and spikes on each cell are generated according to unique Poisson sequence at `spike-frequency` Hz.
