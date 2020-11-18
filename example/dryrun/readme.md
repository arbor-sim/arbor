# Dryrun Example

A miniapp that demonstrates how to use dry-run mode on a simple network
duplicated across `num_ranks`.

It uses the `arb::tile` to build a network of `num_cells_per_rank` cells of type
`arb::cable_cell`. The network is translated over `num_ranks` domains using
`arb::symmetric_recipe`.

Example:
```
num_cells_per_rank = 4;
num_ranks = 2;

gids = {0, ..., 7}

tile connections:
    dest <- src:

    0 <- 1;
    1 <- 3;
    2 <- 5;
    3 <- 7;

symmetric_recipe inferred connections:
    translate tile connections for second doimain:
    dest <- src:

    0 <- 1;
    1 <- 3;
    2 <- 5;
    3 <- 7;

    4 <- 5;
    5 <- 7;
    6 <- 1;
    7 <- 3;

```

The model of the *tile* can be configured using a json configuration file:

```
./bench.exe params.json
```

An example parameter file for a dry-run is:
```
{
    "name": "dry run test",
    "dry-run": true,
    "num-cells-per-rank": 1000,
    "num-ranks": 2,
    "duration": 100,
    "min-delay": 1,
    "depth": 5,
    "branch-probs": [1.0, 0.5],
    "compartments": [100, 2]
}

```

The parameter file for the equivalent MPI run is:
```
{
    "name": "MPI test",
    "dry-run": false,
    "num-cells-per-rank": 1000,
    "duration": 100,
    "min-delay": 1,
    "depth": 5,
    "branch-probs": [1.0, 0.5],
    "compartments": [100, 2]
}

```
These 2 files should provide exactly the same spike.gdf files.


The parameters in the file:
  * `name`: a string with a name for the benchmark.
  * `dry-run`: a bool indicating whether or not to use dry-run mode.
    if false, use MPI if available, and local distribution if not.
  * `num-ranks`: the number of domains to simulate. Only for dry-run
    mode.
  * `num-cells-per-rank`: the number of cells on a single tile.
    The total number of cells in the model = num-cells-per-rank *
    num-ranks.
  * `duration`: the length of the simulated time interval, in ms.
  * `min-delay`: the minimum delay of the network.
  
In addition, these parameters for the synthetic benchmark cell are
understood:
  * `depth`: number of levels, excluding soma (default: 5)
  * `branch-probs`: Probability of a branch occuring (default: 1-0.5).
  * `compartments`: Compartment count on a branch (default: 20-2).
  * `lengths`: Length of branch in μm (default: 200-20).
  * `synapses`: The number of synapses per cell (default: 1).

Parameters given as ranges will take on the first value at the soma
and the second at the leaves, values in between will be interpolated
linearly.

The network is randomly connected with no self-connections, with every
connection having delay of `min-delay`.