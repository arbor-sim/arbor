# Dryrun Example

A miniapp that demonstrates how to use dry-run mode to simulate the effect of communication scaling
with benchmark cells, useful for evaluating the communication overheads associated with
is weak-scaled a model.
The overheads of processing spikes and generating local events do not weak scale perfectly: the cost
of traversing the global spike list increases with global model size.
Whether these overheads contribute significantly to total run time depends on the computational complexity
of the local model and the size of the global model.

## How it works

The dry run mode mimics running a distributed model on a single node or laptop by creating a local model
and generating fake spike information for the other "ranks" in the larger distributed model.

The benchmark allows the user to tune three key parameters
    1. the number of ranks in the global model, which can be used to simulate weak scaling
    2. the size number and computational complexity of cells in the local model
    3. the complexity of the network (fan in and min-delay).

By tuning the parameters above to match those of a target distributed model, it is possible to replicate
the spike and event processing overheads without having to run resource-intensive benchmarks at scale.

**Note**: instead of performing MPI communication to gather the global spike list, the dry run mode
creates a fake global spike list using the local spike data as a template. As such, the scaling of the MPI library
is not captured, which would have to be benchmarked separately if it is a relevant.

## Configuration

The benchmark uses an `arb::tile` to build a network of `num_cells` local cells of type
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
