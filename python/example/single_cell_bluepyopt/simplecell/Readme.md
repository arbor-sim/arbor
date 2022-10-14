# BluePyOpt simple cell output

This example was created in an environment with the `bluepyopt` package installed (see [installation instructions](https://github.com/BlueBrain/BluePyOpt#installation)). A cell model with parameters optimized in [simplecell.ipynb](https://github.com/BlueBrain/BluePyOpt/blob/master/examples/simplecell/simplecell.ipynb) can be exported with


```shell
python /path/to/BluePyOpt/examples/simplecell/generate_acc.py --output <output-dir>
```

An Arbor simulation using this cell model can then be launched using

```shell
python ../../single_cell_bluepyopt_simple.py simple_cell.json
```
