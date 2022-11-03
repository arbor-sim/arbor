# BluePyOpt layer-5 pyramidal cell with axon-replacement

This example was created in an environment with the `bluepyopt` package installed (see [installation instructions](https://github.com/BlueBrain/BluePyOpt#installation)). A cell model with parameters as published by [Markram et al., "Reconstruction and simulation of neocortical microcircuitry", Cell 163.2 (2015): 456–492](http://www.cell.com/abstract/S0092-8674%2815%2901191-5) (see [L5PC.ipynb](https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/L5PC.ipynb)) can be exported with

```shell
python /path/to/BluePyOpt/examples/l5pc/generate_acc.py --output <output-dir> --replace-axon
```

We use the Arbor BBP mechanism catalogue as a substitute for the one in BluePyOpt and an Arbor simulation using this cell model can be launched using

```shell
python ../../single_cell_bluepyopt_l5pc.py l5pc.json
```
