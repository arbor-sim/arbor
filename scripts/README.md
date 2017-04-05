# tsplot

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


# profstats

`profstats` collects the profiling data output from multiple MPI ranks and performs
a simple statistical summary.

Input files are in the JSON format emitted by the profiling code.

By default, `profstats` reports the quartiles of the times reported for each
profiling region and subregion. With the `-r` option, the collated raw times
are reported instead.

Output is in CSV format.


# cc-filter

`cc-filter` is a general purpose line-by-line text processor, with some
built-in rules for simplifying output comprising templated C++ identifiers.

Full documentation for `cc-filter` can be obtained by running it with the
`--man` option. The information below has been transcribed from this output.

In the `filter` subdirectory there is a sample table `massif-strip-cxx`
that will remove the C++ content from the valgrind massif tool output.
This can be then be used without running the default rules with the following:
```
cc-filter -n -t filters/massif-strip-cxx
```

## Options

#### **-n**, **--no-default**

Omit the built-in rules from the default list.

#### **-r**, **--rule=RULE**

Apply the rule or group of rules **RULE**.

#### **-t**, **--table=FILE**

Add the macro, rule and table definitions in **FILE**.

#### **-d**, **--define=DEF**

Add an explicit definition.

#### **-l**, **--list\[=CAT\]**

By default, list the applicable rules and definitions. If **CAT** is
`expand`, expand any macros in the definitions. If **CAT** is
`group`, list the group definitions. If **CAT** is `macro`, list the
macro definitions.

#### **-h**, **--help**

Print help summary and exit.

#### **--man**

Print the full documentation as a man page.

## Description

Rules are applied sequentially to each line of the input files in turn.
The rules are taken from the built-in list, and from any rules defined
in tables supplied by the `--table` option. If the table file is not an
absolute path, it is looked for first in the current directory, and then
relative to the directory in which `cc-filter` resides.

The default list of rules comprises all the rules specified in the
built-in list any supplied table, however no default list is used if a
rules are specifically requested with the `--rule` option. The built-in
rules are omitted from the default list if the `--no-default` option is
given. Rules can be explicitly omitted with the `--exclude` option.

Tables can include groups of rules for ease of inclusion or omission
with the `--rule` or `--exclude` options.

## Table format

Each line of the table is either blank, a comment line prefixed with
'\#', or an entry definition. Definitions are one of three types:
macros, rules, or groups.

### Macros

Macros supply text that is substituted in rule definitions.
A macro definition has the form:
>  `macro` *name* *definition*

The *name* of the macro may not contain any whitespace, and the
*text* of the macro definition cannot begin with whitespace.

Every occurance of `%`*name*`%` in a rule definition will be
substituted with *text*. Macro substitution is recursive: after all
macro substitutions are performed, the rule definition will again be
parsed for macros.

### Rules

A rule definition has the form:
>  `rule` *name* *code*

Rule *name*s may not contain any whitespace.

The *code* entry of a rule undergoes macro expansion (only macros
whose definitions have already been read will apply) and then is
compiled to a perl subroutine that is expected to operate on `$_` to
provide a line transformation.

If a rule is defined multiple times in the same table, the
transformations are concatenated.

If a rule is defined in a subsequent table, the new definition will
replace the old definition.

### Groups

A group definition has the form:
> `group` *name* *rule-or-group-name* … 

Rule (or group) names comprising the definition are separated by
whitespace, and must have already been defined in this or a
previous table.

Definitions added explicitly with the `--define` option are treated as
lines in a table that is parsed after all other tables.

## Example table

Consider a table file `example.tbl` with the lines:

    # a comment comprises a # and any following characters, plus any
    # preceding whitespace.
    macro non-comment (^.*?)(?=\s*(?:#|$))
    rule rev-text s/%non-comment%/$1=~s,[[:punct:]]+,,gr/e
    rule rev-text s/%non-comment%/reverse(lc($1))/e

This defines one rule, `rev-text` which will remove punctuation in the
text preceding a possible comment, and then lower-case and reverse it.

    $ echo 'What, you egg!  # ?!' | cc-filter -n --table example
    gge uoy tahw  # ?!


# PassiveCable.jl

Compute analytic solutions to the simple passive cylindrical dendrite cable
model with step current injection at one end from _t_ = 0.

This is used to generate validation data for the first Rallpack test.

Module exports the following functions:

 * `cable_normalized(x, t, L; tol)`

   Compute potential _V_ at position _x_ in [0, _L_] at time _t_ ≥ 0 according
   to the normalized cable equation with unit length constant and time constant.

   Neumann boundary conditions: _V'_(0) = 1; _V'_(L) = 0.
   Initial conditions: _V_( _x_, 0) = 0.

   Absolute tolerance `tol` defaults to 1e-8.

 * `cable(x, t, L, lambda, tau, r, V, I; tol)`

   Compute the potential given:
      *  length constant `lambda`
      *  time constant `tau`,
      *  axial linear resistivity `r`
      *  injected current of `I` at the origin
      *  reversal potential `V`

   Implied units must be compatible, e.g. SI units.

   Absolute tolerance `tol` defaults to 1e-8.

 * `rallpack1(x, t; tol)`

   Compute the value of the potential in the Rallpack 1 test model at position
   `x` and time `t`.

   Parameters for the underlying cable equation calculation are taken from the
   Rallpack model description in SI units; as the cable length is 1 mm in this
   model, `x` can take values in [0, 0.001].

   Absolute tolerance `tol` defaults to 1e-8.
