# Local field potential demo

How might one use Arbor to compute the local field potential near a cell?

This example provide a simple demonstration, simulating one cell and computing
the LFP from the total membrane current.

The code attempts to provide an Arbor version of the supplied NEURON LFP example
`neuron_lfp_example.py`. The plot from the NEURON code is included as `example_nrn_EP.png`.

## How to run

The example builds an executable `lfp` that performs a 100 ms simulation of
a single ball-and-stick neuron, measuring the local field potential at two
electrode sites.

Running `lfp` generates a JSON file with the simulation output written to stdout.
The included `plot-lfp.py` script will parse this output and generate a plot.

Run `lfp` and display the output in a window:
```
lfp | plot-lfp.py
```

Run `lfp` and save the results, then generate a plot image:
```
lfp > out.json
plot-lfp.py -o out.png out.json
```

