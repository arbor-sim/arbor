tsplot
------

The `tsplot` script is a wrapper around matplotlib for displaying a collection of
time series plots.

## Input data

`tsplot` reads timeseries in JSON format, according to the following conventions.

```
{
    "units": <units>
    "name":  <name>
    <other key-value metadata>
    "data": {
        "time": [ <time values> ]
        <trace name>: [ <trace values> ]
    }
}
```

The `data` object must contain numeric arrays, with at least one with the key `time`;
other members of `data` correspond to traces sampled at the corresponding time values.

The other members of the top level object are regarded as metadata, with some keys
treated specially:
 * `units` are used to distinguish different axes for plotting, and the labels for those
   axes. It's value is either a string, where the specified unit is taken as applying to
   all included traces, or an object representing a mapping of trace names to their
   corresponding unit string.
 * `name` is taken as the title of the corresponding plot, if it is unambiguous.
 * `label` is ignored: the _label_ for a trace is its name in the data object.

## Operation

The basic usage is simply:
```
tsplot data.json ...
```
which will produce an interactive plot of the timeseries provided by the provided
files, with one trace per subplot.

### Grouping

Traces can be gathered on to the same subplot by grouping by metadata with the
`-g` or `--group` option. To collect all traces with the same value of the key
'id' and the same units:
```
tsplot -g units,id data.json ...
```
A subplot can comprise data with to two differint units, and will be plotted
with two differing vertical axes.

Note that for the purposes of tsplot, the value of the key _label_ is the
propertu name of the trace in its json representation.

### Restricting data

The `-t` or `--trange` option exlcudes any points that have a time range outside
that specified. Ranges are given by two numbers separated by a comma, but one or
the other can be omitted to indicate that there is no bound on that side. For
example:
```
tsplot -t ,100 data.json ...
```
will display all points with a time value less than or equal to 100.

Extreme values for data can be automatically excluded and marked on the plot
with the `-x` or `--exclude` option, taking a parameter _N_. All values in a
timeseries that lie outside the interval [ _m_ - _Nr_, _m_ + _Nr_ ] are omitted,
where _m_ is the median of the finite values in the timeseries, and _r_ is
the 90% interquantile gap, that is, the difference between the 5% and 95% quantile
of the timeseries data.

### Output to file

Use the `-o` or `--output` option to save the plot as an image, instead of
displaying it interactively.


