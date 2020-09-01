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

#### **-n**, **--no-default-rules**

Omit the built-in rules from the default list.

#### **-N**, **--no-built-ins**

Omit all rule, group, and macro definitions from the default table.

#### **-r**, **--rule=RULE**

Apply the rule or group of rules **RULE**.

#### **-x**, **--exclude=RULE**

Skip the application of the rule or group of rules **RULE**.

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

Each line has any terminal newline stripped before processing, and then
is subjected to each rule's action in turn via `$_`. If a rule introduces a
newline character, the string is not split for processing by subsequent rules.
(This is a limitation that may be addressed in the future.) If a rule
sets `$_` to `undef`, the line is skipped and processing starts anew with
the next input line.

Tables can include groups of rules for ease of inclusion or omission
with the `--rule` or `--exclude` options.

For details on the table format and example tables, refer to the full
documentation provided by `cc-filter --man`.

## Example usage

The rules applied by default can be listed with `cc-filter --list`, e.g.

```
$ cc-filter --list
cxx:rm-allocator	s/(?:,\s*)?%cxx:qualified%?allocator%cxx:template-args%//g
cxx:rm-delete	s/(?:,\s*)?%cxx:qualified%?default_delete%cxx:template-args%//g
cxx:rm-std	s/%cxx:std-ns%//g
cxx:rm-std	s/%cxx:gnu-internal-ns%//g
cxx:rm-template-space	s/%cxx:template-args%/$1=~s| *([<>]) *|\1|rg/eg
cxx:unsigned-int	s/\bunsigned\s+int\b/unsigned/g
cxx:strip-qualified	s/%cxx:qualified%//g
cxx:strip-args	s/(%cxx:identifier%%cxx:template-args%?)%cxx:paren-args%/$1(...)/g
```

These actions are as follows:

* Remove `allocator<...>` entries from template argument lists (`cxx:rm-allocator`).

* Remove `default_delete<...>` entries from template argument lists (`cxx:rm-delete`).

* Remove `std::` qualifiers (`cxx:rm-std`).

* Remove `__gnu_cxx::` qualifiers (`cxx:rm-std`).

* Collapse spaces between template brackets (`cxx:rm-template-space`).

* Replace occurances of `unsigned int` with `unsigned` (`cxx:unsigned-int`).

* Strip all class or namespace qualifers (`cxx:strip-qualified`).

* Replace argument lists of (regularly named) functions with `(...)` (`cxx:strip-args`).

The rules are grouped, however, so the more invasive transformations can be
straightforwardly enabled or disabled. The defined groups are listed with
`cc-filter --list=group`:

```
cxx:tidy	cxx:rm-template-space cxx:unsigned-int
cxx:strip-all	cxx:strip-qualified cxx:strip-args
cxx:std-simplify	cxx:rm-allocator cxx:rm-delete cxx:rm-std
```

`cc-filter --rule ccx:tidy` would perform only the space and `unsigned` transformations.
`cc-filter --exclude ccx:strip-all` would leave arguments and non-standard namespace and class
qualifications intact while applying the other transformations.

One can see in the rule list the use of some in-built macros, such as `%cxx:template-args%`. These
macro definitions can be listed with `cc-filter --list=macro`:

```
$ cc-filter --list=macro
cxx:std-ns	(?:(::)?\bstd::)
cxx:identifier	(\b[_\pL][_\pL\p{Nd}]*)
cxx:gnu-internal-ns	(?:(::)?\b__gnu_cxx::)
cxx:template-args	(<(?:(?>[^<>]+)|(?-1))*>)
cxx:qualified	(?:(::)?\b(\w+::)+)
cxx:paren-args	(\((?:(?>[^()]+)|(?-1))*\))
```

Rule definitions with macros expanded can be displayed with `--list=expand`, e.g.

```
$ cc-filter --rule cxx:rm-std --list=expand
cxx:rm-std	s/(?:(::)?\bstd::)//g
cxx:rm-std	s/(?:(::)?\b__gnu_cxx::)//g
```

### Built-in rules in action

Consider the following error message generated by g++ (some of the middle lines and the full paths to gcc headers have been elided):

```
In file included from /.../g++/vector:62:0,
                 from badvec.cc:1:
/.../g++/bits/stl_construct.h: In instantiation of 'void std::_Construct(_T1*, _Args&& ...) [with _T1 = long_namespace::bad; _Args = {const long_namespace::bad&}]':
/.../g++/bits/stl_uninitialized.h:75:18:   required from 'static _ForwardIterator std::__uninitialized_copy<_TrivialValueTypes>::__uninit_copy(_InputIterator, _InputIterator, _ForwardIterator) [with _InputIterator = const long_namespace::bad*; _ForwardIterator = long_namespace::bad*; bool _TrivialValueTypes = false]'
[...]
/.../g++/bits/stl_vector.h:379:2:   required from 'std::vector<_Tp, _Alloc>::vector(std::initializer_list<_Tp>, const allocator_type&) [with _Tp = long_namespace::bad; _Alloc = std::allocator<long_namespace::bad>; std::vector<_Tp, _Alloc>::allocator_type = std::allocator<long_namespace::bad>]'
badvec.cc:10:48:   required from here
/.../g++/bits/stl_construct.h:75:7: error: use of deleted function 'long_namespace::bad::bad(const long_namespace::bad&)'
     { ::new(static_cast<void*>(__p)) _T1(std::forward<_Args>(__args)...); }
       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
badvec.cc:6:5: note: declared here
     bad(const bad&) = delete;
     ^~~
```

Running this message through `cc-filter`:


```
In file included from /.../g++/vector:62:0,
                 from badvec.cc:1:
/.../g++/bits/stl_construct.h: In instantiation of 'void _Construct(...) [with _T1 = bad; _Args = {const bad&}]':
/.../g++/bits/stl_uninitialized.h:75:18:   required from 'static _ForwardIterator __uninitialized_copy<_TrivialValueTypes>::__uninit_copy(...) [with _InputIterator = const bad*; _ForwardIterator = bad*; bool _TrivialValueTypes = false]'
[...]
/.../g++/bits/stl_vector.h:379:2:   required from 'vector<_Tp, _Alloc>::vector(...) [with _Tp = bad; _Alloc = ; vector<_Tp, _Alloc>::allocator_type = ]'
badvec.cc:10:48:   required from here
/.../g++/bits/stl_construct.h:75:7: error: use of deleted function 'bad(...)'
     { ::new(...) _T1(...); }
       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
badvec.cc:6:5: note: declared here
     bad(...) = delete;
     ^~~
```

If we wanted to retain the function call arguments, run with `cc-filter --exclude ccx:strip-args`:


```
In file included from /.../g++/vector:62:0,
                 from badvec.cc:1:
/.../g++/bits/stl_construct.h: In instantiation of 'void _Construct(_T1*, _Args&& ...) [with _T1 = bad; _Args = {const bad&}]':
/.../g++/bits/stl_uninitialized.h:75:18:   required from 'static _ForwardIterator __uninitialized_copy<_TrivialValueTypes>::__uninit_copy(_InputIterator, _InputIterator, _ForwardIterator) [with _InputIterator = const bad*; _ForwardIterator = bad*; bool _TrivialValueTypes = false]'
[...]
/.../g++/bits/stl_vector.h:379:2:   required from 'vector<_Tp, _Alloc>::vector(initializer_list<_Tp>, const allocator_type&) [with _Tp = bad; _Alloc = ; vector<_Tp, _Alloc>::allocator_type = ]'
badvec.cc:10:48:   required from here
/.../g++/bits/stl_construct.h:75:7: error: use of deleted function 'bad(const bad&)'
     { ::new(static_cast<void*>(__p)) _T1(forward<_Args>(__args)...); }
       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
badvec.cc:6:5: note: declared here
     bad(const bad&) = delete;
     ^~~
```


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
