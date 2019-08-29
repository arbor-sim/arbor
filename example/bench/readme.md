# Artificial Infrastructure Benchmark

This miniapp uses the `arb::benchmark_cell` to build a network of cells that
have predictable cell integration run time and spike generation patterns. It is
for benchmarking purposes only.

The main application of this miniapp is to benchmark the Arbor simulation
architecture. The architecture is everything that isn't cell state update,
including
  * Spike exchange.
  * Event sorting and merging.
  * Threading and MPI infrastructure.

For example, the scaling behavior of the spike exchange can be studied as
factors such as fan-in, network minimum delay, spiking frequency and spiking
pattern are varied, without having to tweak parameters on a model that uses
"biologically realistic" cells such as LIF cells.

## Usage

The model can be configured using a json configuration file:

```
./bench params.json
```

An example parameter file is:
```
{
    "name": "small test",
    "num-cells": 2000,
    "duration": 100,
    "fan-in": 10000,
    "min-delay": 10,
    "spike-frequency": 20,
    "realtime-ratio": 0.1
}
```

The parameters in the file:
  * `name`: a string with a name for the benchmark.
  * `num-cells`: the total number of cells in the model. The cell population
    is assumed to be homogoneous, that is the `spike-frequency` and
    `cell-overhead` parameters are the same for all cells.
  * `duration`: the length of the simulated time interval, in ms.
  * `fan-in`: the number of incoming connections on each cell.
  * `min-delay`: the minimum delay of the network.
  * `spike-frequency`: frequency of the independent Poisson processes that
    generate spikes for each cell.
  * `realtime-ratio`: the ratio between time taken to advance a single cell in
    the simulation and the simulated time. For example, a value of 1 indicates
    that the cell is simulated in real time, while a value of 0.1 indicates
    that 10s can be simulated in a single second.

The network is randomly connected with no self-connections and `fan-in`
incoming connections on each cell, with every connection having delay of
`min-delay`.
