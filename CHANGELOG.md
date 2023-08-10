# v0.9.0

** 2023 08 09 **

After much more delay than anticipated, we are very happy to present a new Arbor release. Nearly 8 months of work is in it, much of which focussed on speed, optimisation, fixes and build system changes. This release includes Python 3.12 wheels as probably one of the first packages on PyPI 😄.

## Major new features

* External Connectivity: Arbor can now interface with other simulators or processes through MPI. Contexts now accept a second MPI communicator and Recipes can implement a ``external_connections_on`` method to specify how simulations might be interacting. [See documentation for more.](https://docs.arbor-sim.org/en/latest/concepts/interconnectivity.html#cross-simulator-interaction)
* Arbor now supports checkpointing and resume from a previously stored checkpoint. Useful when a certain point in time needs to be explored in multiple directions, when you want to rewind to a previous state, etc. [See documentation for more.](https://docs.arbor-sim.org/en/latest/fileformat/serdes.html)
* More painted parameters are now scalable through `iexpr`: temperature, capacitance, resistivity, membrane potential and the following ionic parameters: internal and external concentration, diffusivity, and reversal potential. [See documentation.](https://docs.arbor-sim.org/en/latest/concepts/decor.html#scaling-mechanism-and-membrane-parameters)
* A set of ANN activation functions for NMODL have been added: `sigmoid(x)`, `relu(x)` and `tanh(x)`. Control volume `area` is exposed through NMODL.
* A revamped backend for ARM CPU's that support Scalable Vector Extension (SVE). Arbor and modcc are now fully compatible, so users who have access to A64FX-based HPC can take full advantage of that hardware.

## Breaking changes since v0.8.1

* It is no longer possible to set binning and sampling policies, due to Arbor now having a fixed timestep. Removing exact delivery increases the speed of the simulation due to elimination of small steps, makes the numerics independent of presence of sampling, and also leads to a number of code simplifications.
* Contexts are now constructed kwargs-only. So, `arbor.context(12, None)` is now `arb.context(threads=12, gpu=None)`. In the case no arguments are supplied, the context initialized to `default_allocation`, which means it will use the total number of threads and first GPU that are locally available. `arbor.simulation` constructed without an explicit `arbor.context` also initializes a `default_allocation`.
* Version bumps:
  * CUDA 11 or higher is required for GPU builds.

## Full commit log

### Neuroscience, documentation

* Emit better warning. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2071
* ANN activation functions for NMODL: `sigmoid(x)`, `relu(x)` and `tanh(x)` by @llandsmeer in https://github.com/arbor-sim/arbor/pull/2052
* 📊  Make profiler output a bit nicer. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2078
* Merge GUI tutorial by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2088
* Busyring example by @boeschf in https://github.com/arbor-sim/arbor/pull/2100
* Feature: External Connectivity by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2001
* FEATURE: fixed dt by @boeschf in https://github.com/arbor-sim/arbor/pull/2053
* mention Matrix, GUI, Playground in docs by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2113
* Document presets by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2126
* Doc updates by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2119
* CV area exposed to NMODL. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2110
* Fix enumeration in docs by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2140
* Docs: corrections. by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2135
* Fix documentation  by @boeschf in https://github.com/arbor-sim/arbor/pull/2152
* Address missing logo on all but front page by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2155
* Document features by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2158
* Make (most) painted parameters scalable. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2141
* Add better (helpful?) errors on mechanism semantic misuse. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2171
* Checkpoint/Resume by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2081

### Core

* Better handling of powers in modcc. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2061
* Silence spurious unused variable warnings in generated code by @bcumming in https://github.com/arbor-sim/arbor/pull/2073
* BluePyOpt API updates by @lukasgd in https://github.com/arbor-sim/arbor/pull/2045
* Add some missing #include directives by @musicinmybrain in https://github.com/arbor-sim/arbor/pull/2080
* Remove redundant sample events structure. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2079
* 🧹 Re-factor FVM lowered cell implementation and shared state by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2082
* Take less locks for shorter times when sampling. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2093
* Fix LIF oversampling. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2091
* 📊 Pin threads by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2094
* Add zero'ing by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2107
* Add an error on NET_RECEIVE w/ > 1 args. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2127
* Update busyring to generate rings by @bcumming in https://github.com/arbor-sim/arbor/pull/2144
* Apply some polish to model construction. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2134
* SVE backend by @boeschf in https://github.com/arbor-sim/arbor/pull/2148
* Exit `sup::path` stage left; enter `std::filesystem::path` stage right by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2165
* gtest dependency by @boeschf in https://github.com/arbor-sim/arbor/pull/2163

### Build, testing, CI

* v0.8.1 by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2072
* v0.8.1 postrelease by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2074
* Enable Spack tests by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2098
* Python/pip strikes again: execute `pip install` in venvs by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2111
* Scikit build core by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2121
* Spack: add dev version by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2150
* Break out pip and doc tests, exclude doc-change from eating up precious runner cycles by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2160
* rename test-pip.yml by @boeschf in https://github.com/arbor-sim/arbor/pull/2166
* cscs-ci by @boeschf in https://github.com/arbor-sim/arbor/pull/2149
* gtest in CI containers by @boeschf in https://github.com/arbor-sim/arbor/pull/2170
* Spack-solvable versions, add tests to Spack workflow by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2156
* Fudge version in pyproject.toml for ci by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2200
* enable remote example by @boeschf in https://github.com/arbor-sim/arbor/pull/2173

### Fixes, optimization

* [BUGFIX] a-b-c: actually set compiler, improved default by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2051
* 🐙 Optimise PPACK by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2067
* Silence flake8 and appease black. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2086
* BUGFIX: add missing functions to sve backend by @boeschf in https://github.com/arbor-sim/arbor/pull/2096
* BUGFIX gpu ppack declarations by @boeschf in https://github.com/arbor-sim/arbor/pull/2087
* BUGFIX: gpu mechanism init by @boeschf in https://github.com/arbor-sim/arbor/pull/2095
* The Initialisation Optimisation by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2097
* Fix crash in FindId Visitor. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2115
* Add LTO and M1 arch to CMake by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2106
* Fix crash on probing non-existent mechanism state. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2112
* Python/concurrency defaults by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1979
* Add faster merge events. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2108
* Fix a typo in python bindings. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2125
* Fix incorrect mention of attribute `catalogue` in `single_cell_model` by @Helveg in https://github.com/arbor-sim/arbor/pull/2122
* Avoid crash when integrating over empty branch. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2124
* fix lint pipleine by @boeschf in https://github.com/arbor-sim/arbor/pull/2151
* Ward against probing the void. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2139
* Fix GPU/NVC problems by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2143
* fix ODR violation by @boeschf in https://github.com/arbor-sim/arbor/pull/2154

## New Contributors
* @musicinmybrain made their first contribution in https://github.com/arbor-sim/arbor/pull/2080

**Full Changelog**: https://github.com/arbor-sim/arbor/compare/v0.8...v0.9.0-rc


# v0.8.1

** 2022 12 22 **

A 🎄 holiday release! Not much has changed in a month, but we'd like to share it all the same. Notably, the [Arbor GUI](https://github.com/arbor-sim/gui/) [is co-released](https://github.com/arbor-sim/gui/releases/tag/v0.8) as of Arbor v0.8, and v0.8.1 will be no different.

## Major new features

- Voltage Processes: add the VOLTAGE_PROCESS mechanism kind to modcc, allowing for direct writing to the membrane voltage (#2033)
- Spack gpu option: added conditional variant for cuda builds to enable GPU-based random number generation (#2043) 
- SDE Tutorial (#2044) 

## Breaking changes since v0.8

- None 💃!

## Bug fixed

- Fix ornstein_uhlenbeck example on gpu (#2039)
- Setting ARB_MODCC was broken and nunfunctional. Fixed. (#2029)
- The `--cxx` flag in `arbor-build-catalogue` is now properly used; falls back to `c++`. (#2051)

## Full commit log

* Post release: add Zenodo entry, add Spack entry, update docs and scripts by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2036
* BUGFIX: add ARB_CUDA flag to example catalogue by @boeschf in https://github.com/arbor-sim/arbor/pull/2039
* Additional builtin functions to Arbor's NMODL dialect by @boeschf in https://github.com/arbor-sim/arbor/pull/2035
* Throw better errors when we cannot look up ion diffusivity by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2040
* ⚡ Voltage Processes by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2033
* simplify make catalogue by @boeschf in https://github.com/arbor-sim/arbor/pull/2042
* spack gpu option by @boeschf in https://github.com/arbor-sim/arbor/pull/2043
* Remove deprecated spike_detector. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2041
* make ARB_MODCC functional again by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2029
* Yank v0.5 from Spack file, it does not build due to change in setting arch by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2037
* Use pugixml instead of libxml2 by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2048
* [BUGFIX] a-b-c: actually set compiler, improved default by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2051
* Sde Tutorial by @boeschf in https://github.com/arbor-sim/arbor/pull/2044
* Set Python to known version for all CI workflows by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2058
* consistent mechanism ids by @boeschf in https://github.com/arbor-sim/arbor/pull/2057
* Fix docs about exprelr. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2064
* Allow __ in profiler names. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2065
* CI fixes for wheel building, prep for musllinux, Spack by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2069

# v0.8

** 2022 11 15 **

Welcome to another installment of the Arbor simulator!

In this release we add a CHANGELOG, where major new features and breaking changes will be mentioned specifically.

## Major new features

- Stochastic Differential Equations. Introduces sources of white noise to arbor (and nmodl). Both point and density mechanisms may now use white noise as part of the state updates, turning the ODEs effectively into SDEs. The noise sources are provided per connection end point (point mech.) or control volume (density mech.) and are uncorrelated (spatially, temporally and across different mechanism instantiations). https://github.com/arbor-sim/arbor/pull/1884
- Mutable connection table. Add functionality, docs, and examples on editing the connection table. This is a first, small PR
on the topic, further functionality will come as requested. https://github.com/arbor-sim/arbor/pull/1919
- Allow editing morphologies. Supported operations: join_at, split_at, equivalence, equality, apply isometry. https://github.com/arbor-sim/arbor/pull/1957
- Arbor cable cell exporter and backend support in BluePyOpt. https://github.com/arbor-sim/arbor/pull/1959
- Make LIF cells probeable. https://github.com/arbor-sim/arbor/pull/2021

## Breaking changes since v0.7:

- A change in decor API: `arbor.cable_cell` has the labels and decor arguments swapped. I.e.: `(tree, labels, decor)`
-> `(tree, decor, label)`. Labels are now optional. https://github.com/arbor-sim/arbor/pull/1978
- Remove the `generate-catalogue` script.  `modcc` accepts now a list of NMODL files and is able to spit out a catalogue.cpp file. https://github.com/arbor-sim/arbor/pull/1975
- Mechanism ABI version is bumped to 0.3.1. https://github.com/arbor-sim/arbor/pull/1884
- Rename spike detector -> threshold detector. https://github.com/arbor-sim/arbor/pull/1976
- Remove access to time `t` in NMODL. https://github.com/arbor-sim/arbor/pull/1967
- Major dependency version bumps:
  - GCC: 9 and up
  - CUDA: 11 and up
  - Clang: 10 and up

## Full commit log

### Neuroscience, documentation

* Convenience: shorter code to build decor. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1903
* Fix doc error. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1934
* probe id -> probeset id, and clarification in docs by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1898
* Elaborate on mpi4py. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1940
* Add all_closest to place_pwlin. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1952
* Add more visibility to decor. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1953
* add spike source docs. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1939
* 🦑  Excise `time` by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1967
* 🐍  Rename spike detector -> threshold detector by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1976
* Mutable connection table by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1919
* Allow multiple schedules per source_cell. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1963
* Allow editing morphologies by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1957
* generate-catalogue is gone by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1975
* SDE by @boeschf in https://github.com/arbor-sim/arbor/pull/1884
* Discuss q10 pattern in NMODL docs. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1995

### Core

* modcc: Allow redundant, but correct READ declaration. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1936
* Deny WATCH statements. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1942
* Excise fvm type aliases by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1938
* Add stack information to `arbor_exception` by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1945
* Expose find_private_gpu to python by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1943
* Make `mpoint`s hashable. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1950
* Arborio reads from Stream objects now. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1937
* Remove explicit generator by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1962
* Clean up plasticity by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1985
* Heaviside step by @llandsmeer in https://github.com/arbor-sim/arbor/pull/1989
* Do not restrict SWC record identifier by @schmitts in https://github.com/arbor-sim/arbor/pull/1996
* Make decor mandatory and labels optional. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1978
* Add virtual dtors to recipe components by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2000
* cleanup documentation by @boeschf in https://github.com/arbor-sim/arbor/pull/2007
* Arbor cable cell exporter and backend support in BluePyOpt by @lukasgd in https://github.com/arbor-sim/arbor/pull/1959
* Add a plethora of config options to a-b-c. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1958
* Make LIF cells probeable. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2021
* Added fixture dev docs. Made fixtures more robust. by @Helveg in https://github.com/arbor-sim/arbor/pull/2025
* Make LIF cells LIF cells again. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2026
* update cray documentation by @boeschf in https://github.com/arbor-sim/arbor/pull/2022

### Build, testing, CI

* Spack cache change, bump versions by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1926
* Expand docs on testing. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1944
* Fix cmake paths so we can use Arbor as a sub-project. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1948
* bump pybind11 for py3.11 compat by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1955
* Add spike counts to pre-commit tests. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1965
* Move pybind11 to `/ext` by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1968
* Add black config to pyproject.toml. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1971
* Bump CI workflow and GoogleTest by @boeschf in https://github.com/arbor-sim/arbor/pull/2003
* Remove cscs-ci badges by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1999
* fix cmake: ninja + ExternalProject by @boeschf in https://github.com/arbor-sim/arbor/pull/2008
* Use HTTPS access instead of ssh/git for gtest. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2009
* Add memory sanitizer by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2013
* Try to use OpenMPI as of brew/apt. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2016
* Bump versions by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2017
* Include ubenches into CI by @boeschf in https://github.com/arbor-sim/arbor/pull/2014
* Speed up CI by @boeschf in https://github.com/arbor-sim/arbor/pull/2019
* Update pyproject.toml for Py3.11 by @brenthuisman in https://github.com/arbor-sim/arbor/pull/2018
* Check internal invariants in CI by @thorstenhater in https://github.com/arbor-sim/arbor/pull/2023

### Fixes, optimization

* Disable inconsistent rule. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1928
* Action must change `VERSION` except on version tag. by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1931
* NMDOL: default catalogue clean up by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1935
* PANIC! Forgot to fix fvm type in GPU!!!. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1946
* Remove abuse of arg_v. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1954
* Catch symdiff errors. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1964
* Bug fix: point mechs applying weights to ionic concentrations by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1960
* Fix some issues found by PVS Studio. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1974
* load_asc Python fix  by @lukasgd in https://github.com/arbor-sim/arbor/pull/1977
* Ensure proper Pybind11 is found. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1961
* missing std namespace by @boeschf in https://github.com/arbor-sim/arbor/pull/1994
* Optimize CPU-side solvers by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1992
* 🦑 modcc now generates GPU mechs again. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1988
* 🦑 Never call a procedure again. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1972
* compiler warnings by @boeschf in https://github.com/arbor-sim/arbor/pull/2006
* guard for zero random variables by @boeschf in https://github.com/arbor-sim/arbor/pull/2031
* fix some compiler warning by @boeschf in https://github.com/arbor-sim/arbor/pull/2034

## New Contributors
* @lukasgd made their first contribution in https://github.com/arbor-sim/arbor/pull/1977

**Full Changelog**: https://github.com/arbor-sim/arbor/compare/v0.7...v0.8-rc



# v0.7

** 2022 07 20 **

## What's Changed

### Neuroscience/documentation
* prep extracellular potentials tutorial. Updated corresponding example by @espenhgn in https://github.com/arbor-sim/arbor/pull/1825
* Add documentation on faster NMODL. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1840
* Add tstop doc in recipe API doc by @schmitts in https://github.com/arbor-sim/arbor/pull/1852
* Use list comprehension to speed up creation of connections by @schmitts in https://github.com/arbor-sim/arbor/pull/1864
* Support markers in neurolucida ascii files by @bcumming in https://github.com/arbor-sim/arbor/pull/1867
* simplified `create_polygon` function in lfpykit example by @espenhgn in https://github.com/arbor-sim/arbor/pull/1881
* Better key for the `lfpykit` intersphinx mapping by @Helveg in https://github.com/arbor-sim/arbor/pull/1878
* Sort (in python) the mechanism names (for convenience). by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1882
* Always emit weight expression. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1875
* Add support for epoch callbacks by @bcumming in https://github.com/arbor-sim/arbor/pull/1873
* Add introspection for global properties. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1890
* Add point probes to demo by @schmitts in https://github.com/arbor-sim/arbor/pull/1891
* Tut/use avail threads by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1900
* Allen tutorial by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1781
* New policy 'round_robin_halt' by @jlubo in https://github.com/arbor-sim/arbor/pull/1868
* Axial Diffusion by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1729
* Add some convenience to simulation creation. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1904
* Predefine SWC Regions. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1911
* Diffusion Example Improvements (and a bit of clean-up) by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1914
* Inhomogeneous parameters by @AdhocMan in https://github.com/arbor-sim/arbor/pull/1887
* Fix line numbers in tutorials and assorted doc corrections by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1917

### Core
* 🐍 Be more lenient when accepting args to file I/O by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1819
* modcc: generate missing `node_index` read needed for reading time `t` in the mechanisms by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1866
* Add Developer Documentation by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1639
* Isolate external catalogues from libarbor.a. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1837

### Build, testing, CI
* Build Python 3.10 binary wheels. Add v0.6 to spackfile. by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1817
* export API by @boeschf in https://github.com/arbor-sim/arbor/pull/1824
* export doc by @boeschf in https://github.com/arbor-sim/arbor/pull/1849
* Include CMAKE+CUDA iff NVCC is needed. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1855
* Bit more on Spack, fix in tutorial by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1838
* json submodule added by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1871
* Fix a bug where Debian/Ubuntu's Python malfunctions by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1894
* Have dependency version policy by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1865
* random123 submodule added by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1872
* Fix tool installation paths. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1905
* Adopt Black for Python. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1906
* add cmake checks for non-bundled random123 by @bcumming in https://github.com/arbor-sim/arbor/pull/1907
* Temporarily disable A64FX CI by @bcumming in https://github.com/arbor-sim/arbor/pull/1910
* Adopt flake8 by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1908
* Move Python build to `pyproject.toml`, bump Python minver to 3.7, fix macos wheel generation by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1916
* Weekly CI Python wheel build pushes to Test.PyPI.org by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1921

### Fixes, optimization
* Users may not give dt < 0. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1821
* Fix `ubenches` compilation errors by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1828
* death test: wrong signal by @boeschf in https://github.com/arbor-sim/arbor/pull/1858
* Make brunel.py setup faster. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1854
* add a better error. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1846
* Fix load_component for label-dict by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1859
* Found out the hard way that this is still needed :/ by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1860
* Elide memcpy where not needed by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1863
* Bug fix: Fix voltage vector size in threshold_watcher contstructor by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1820
* remove test statements for move ctor by @boeschf in https://github.com/arbor-sim/arbor/pull/1899
* Code Quality: PVS Studio Finds by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1901
* Two context decomp swaps were forgotten by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1912

## New Contributors
* @jlubo made their first contribution in https://github.com/arbor-sim/arbor/pull/1868
* @AdhocMan made their first contribution in https://github.com/arbor-sim/arbor/pull/1887

**Full Changelog**: https://github.com/arbor-sim/arbor/compare/v0.6...v0.7



## v0.6

** 2022 01 26 **

## What's Changed

Core API:

* Add S-Exp for CV Policies by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1511
* Clean-up and extend mech cat iterator interface. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1564
* Mech ABI: The final step.  by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1452
* Add `dim3` to gridDim constructor in generated mechanism code by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1630
* Added an empty catalogue by @Helveg in https://github.com/arbor-sim/arbor/pull/1677
* Allow to limit Poisson schedule with a stop time (closes #1617) by @schmitts in https://github.com/arbor-sim/arbor/pull/1684
* Expose profiler to Python by @Helveg in https://github.com/arbor-sim/arbor/pull/1688
* Added a debug mode to `build-catalogue` by @Helveg in https://github.com/arbor-sim/arbor/pull/1686
* Write build-catalogue (c)make errors to `stdout` and `stderr` by @Helveg in https://github.com/arbor-sim/arbor/pull/1679
* Makejobs arg by @Helveg in https://github.com/arbor-sim/arbor/pull/1673
* Feature/python clean memory by @max9901 in https://github.com/arbor-sim/arbor/pull/1670
* Feature/label dict by @apeyser in https://github.com/arbor-sim/arbor/pull/1711
* Add locset expressions for proximal and distal translation of locations by @bcumming in https://github.com/arbor-sim/arbor/pull/1671
* Gap Junction mechanisms by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1682
* Solve non-linear systems that are not kinetic schemes. by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1724
* Add cos and sin SVE implementations by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1744
* Deal with zero radius points in a morphology by @halfflat in https://github.com/arbor-sim/arbor/pull/1719
* Simplify default proc_allocation generation. by @halfflat in https://github.com/arbor-sim/arbor/pull/1725
* Find morphology location from a coordinate by @bcumming in https://github.com/arbor-sim/arbor/pull/1751
* resurrect lmorpho by @schmitts in https://github.com/arbor-sim/arbor/pull/1746
* Expose information about the CV discretization to the public inteface by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1758
* Allow the use of string s-expression CV-policies in pyarb  by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1799
* Construct `domain_decomposition` given a list of `group_descriptions`. by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1788
* Profile externally loaded mechanisms by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1691
* Have option to set thread count to local maximum by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1716

CI/build/testing:

* Python unit test clean-up by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1595
* Add CI for the in-repo Spack package by @schmitts in https://github.com/arbor-sim/arbor/pull/1544
* Ci/sanitize by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1521
* Update spack package to include fmt for arbor@0.5.3: by @bcumming in https://github.com/arbor-sim/arbor/pull/1609
* Cache Spack for GHA by @schmitts in https://github.com/arbor-sim/arbor/pull/1619
* Run examples via scripts (closes #1566) by @schmitts in https://github.com/arbor-sim/arbor/pull/1631
* Build a catalogue in CI by @schmitts in https://github.com/arbor-sim/arbor/pull/1632
* Switch to PyPA's `cibuildwheel` action repo's. by @Helveg in https://github.com/arbor-sim/arbor/pull/1703
* Automatic test discovery sans boilerplate by @Helveg in https://github.com/arbor-sim/arbor/pull/1693
* build external mechanism catalogues in release mode by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1713
* `make VERBOSE=1` in verbose mode by @Helveg in https://github.com/arbor-sim/arbor/pull/1715
* Bump pybind to 2.8.1 by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1742
* Better error reporting in python context interface by @bcumming in https://github.com/arbor-sim/arbor/pull/1732
* Better error diagnostic in python when invalid cell properties are provided. by @bcumming in https://github.com/arbor-sim/arbor/pull/1743
* Test separately built catalogues by @bcumming in https://github.com/arbor-sim/arbor/pull/1748
* Weekly cron job checking for submodule updates by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1741
* Added local Python testsuite instructions by @Helveg in https://github.com/arbor-sim/arbor/pull/1809
* CMake CUDA clean-up. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1804
* Add dry run benchmark cell model for testing communication scaling by @bcumming in https://github.com/arbor-sim/arbor/pull/1627
* v0.6-rc by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1814
* More robust build-catalogue by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1784

Documentation:

* `arbor.mechanism_catalogue.extend()` documentation added. by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1597
* zenodo 0.5.2 entry by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1598
* mpi.py updated for new synapse labelling by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1602
* Tutorial structure: remove deduplication, includify by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1575
* Docs mechabi by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1610
* Improved documentation about weights by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1620
* Adapt to changes in Pandas requiring unique indices by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1625
* Add implies default license by @kanzl in https://github.com/arbor-sim/arbor/pull/1635
* Update Slack to Gitter by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1637
* Better Python installation instructions by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1622
* Add example based on LFPykit example Example_Arbor_swc.ipynb by @espenhgn in https://github.com/arbor-sim/arbor/pull/1652
* Name and describe arguments to constructors where ambiguous. by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1678
* improved Github Issue templates by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1698
* Add some comments about units to network ring examples by @schmitts in https://github.com/arbor-sim/arbor/pull/1697
* Fix link (closes #1664) by @schmitts in https://github.com/arbor-sim/arbor/pull/1701
* Fix class members in cable example by @schmitts in https://github.com/arbor-sim/arbor/pull/1641
* Correctify and seabornize dendrite tutorial by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1706
* Polish GJ example readme by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1726
* Completed the docs with more `config()` flags by @Helveg in https://github.com/arbor-sim/arbor/pull/1708
* Make connectivity example more complete by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1721
* Add description of arbor and NEURON's  `nernst` application rules to the docs. by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1638
* Show docs version info more prominently on RTD by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1737
* Update/fix tinyopt README.md by @halfflat in https://github.com/arbor-sim/arbor/pull/1747
* Add python gap junction example by @kanzl in https://github.com/arbor-sim/arbor/pull/1750
* Documentation: add expected nmodl units  by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1756
* Fixed broken link from `get_probes` to `probes` by @Helveg in https://github.com/arbor-sim/arbor/pull/1763
* Removed double preposition by @Helveg in https://github.com/arbor-sim/arbor/pull/1762
* Fixed MD to RST typo by @Helveg in https://github.com/arbor-sim/arbor/pull/1760
* Fix broken links in `probe_sample.rst` by @Helveg in https://github.com/arbor-sim/arbor/pull/1761
* Docs/contrib updates by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1736
* Add Release procedure to docs by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1738
* Example and tutorial for two cells connected via a gap junction by @schmitts in https://github.com/arbor-sim/arbor/pull/1771
* Have single source for citation info, compatible with Github citation UI by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1791
* Advertise GH Discussions for modelling by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1793
* Add release cycle info to docs by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1798
* Added a warning when users are reading latest docs.  by @Helveg in https://github.com/arbor-sim/arbor/pull/1800

Fixes/optimization:

* Fix GPU compile. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1601
* Compile all arbor source files with `-fvisibility=hidden` by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1599
* Bug/assorted static analysis by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1615
* Fix ambiguous Region/Locset expressions by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1629
* Clean-up AVX routine and global def [EOM] by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1618
* Categorise ASSIGNED RANGE variables as STATE-ish. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1624
* Check that mechanisms have the right kind by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1633
* Resolve uninitialised values. by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1616
* Correctly parse diameter in ASCII files by @bcumming in https://github.com/arbor-sim/arbor/pull/1640
* Attempt fixing a64fx build by @bcumming in https://github.com/arbor-sim/arbor/pull/1636
* Fix simulation::add_sampler cell_group dispatch by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1654
* Fix on-components on empty cable bug by @halfflat in https://github.com/arbor-sim/arbor/pull/1658
* Fix cv_policy parse and write in ACC by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1665
* Fixed MRO and code duplication in `setup.py` by @Helveg in https://github.com/arbor-sim/arbor/pull/1672
* Rephrase exception message in case of missing segment by @schmitts in https://github.com/arbor-sim/arbor/pull/1659
* Fix modcc simd generation by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1681
* Segfault on instantiating mechanisms on empty regions by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1657
* Add missing closing bracket to mechanism repr (closes #1667) by @schmitts in https://github.com/arbor-sim/arbor/pull/1699
* Better handling of errors during dynamic catalogue loading by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1702
* Swap && and || operator precedence in modcc by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1710
* Remove use of removed INSTRUMENT_MALLOC symbols from Glibc 2.34 by @brenthuisman in https://github.com/arbor-sim/arbor/pull/1730
* Generate correct simd code for reading `peer_index` and `v_peer` by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1735
* fixes for non-standard code by @boeschf in https://github.com/arbor-sim/arbor/pull/1769
* Modcc: Forward declare procedure kernels in generated GPU code by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1787
* modcc: exit solvers early if errors encountered. by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1755
* Fix cmake configuration of out-of-core fmt library by @bcumming in https://github.com/arbor-sim/arbor/pull/1796
* Bug fix: properly partition networks containing one-sided gap-junction connections by @noraabiakar in https://github.com/arbor-sim/arbor/pull/1774
* Remove footgun: Catalogue Lifetimes by @thorstenhater in https://github.com/arbor-sim/arbor/pull/1807


## New Contributors
* @kanzl made their first contribution in https://github.com/arbor-sim/arbor/pull/1635
* @max9901 made their first contribution in https://github.com/arbor-sim/arbor/pull/1670
* @boeschf made their first contribution in https://github.com/arbor-sim/arbor/pull/1769

**Full Changelog**: https://github.com/arbor-sim/arbor/compare/v0.5.2...v0.6



# v0.5.2

** 2021 06 24 **

This release fixes an error in the CI generated Python wheels, which are as of this release available on PyPI. Other than those fixes, this release is identical to v0.5.1.



# v0.5.1

** 2021 06 22 **

Since v0.5 there have been some major features, and many small fixes and improvements.

Core API features:

* [C++/Python] Labels instead of indices for placeable item identification.
* [C++/Python] Morphology file format support: Arbor Cable-Cell Format.
* [C++/Python] Morphology file format support: Neurolucida ASCII format.
* [C++/Python] Morphology file format improvements: SWC.
* [C++/Python] Simplified connections and junctions.
* [C++/Python] Enable simulation resume/restart.
* [C++/Python] Add post events functionality to support models with STDP synapses
* [C++/Python] Allow dynamically creating and loading of mechanism catalogue

Documentation:

* Documentation URL changed to docs.arbor-sim.org
* New Python examples and tutorials
* Ever more complete documentation
* Added Code of Conduct

Build / CI:

* Spack distribution
* CI generated binary Python wheels
* Apple M1 compatibility
* CI moved from Travis to Github Actions
* Improved Python and C++ unit testing
* ARM CI
* Sanitizer CI

Contributions by, in no specific order, @haampie, @clinssen, @espenhgn, @Helveg, @brenthuisman, @noraabiakar,
@thorstenhater, @halfflat, @schmitts and @bcumming



# v0.5

** 2021 01 07 **

Since v0.4 there have been some major features, and many small fixes and improvements.

Core API features:
* [C++/Python] Numerous small bug fixes, optimizations and improvements.
* [C++/Python] Refactor cable cell interface to be read only, and be constructed from                                       
  descriptions of morphology, labels, and decorations.
* [C++/Python]Expose diverse probes and rich interface for describing where and 
  what to sample on cable cells.
* [C++/Python] Support for querying names in mechanism catalogues
* [Python] Wrapper for existing C++ `pw_lin` functionality
* [C++] Improved validation of recipe definitions during model building

Documentation:
* Added new Python examples
* Many small fixes for links, spelling, grammar and clarity.
* Add extensive guide for contributions and coding polices.

Build:
* Allow CMake configuration to use system copies of C++ dependencies
  (nlohmann/json and pybind11), and makes this the default option.
* Added GitHub Actions support for automated testing of a wider range of tests
  and features than are run on our Travis CI (which will be removed soon)
* More robust Python detection and consistent use of the same Python
  interpreter in CMake configure and build steps.

Contributions by, in no specific order, @brenthuisman, @noraabiakar,
@thorstenhater, @halfflat, @schmitts and @bcumming



# v0.4

** 2019 10 15 **

## Release notes

### Library
* Moved from C++14 to C++17
    * Removed our hand-rolled versions of `any`, `optional` and `variant`.
* Added `std::expected` equivalent for error handling.

### Features
* Added mechanism catalogues with mechanisms used by Allen and BBP models.
* Removed support for spherical segments at the root of cable morphologies, and
  replaced the sample-based representation with a segment-based representation:
    * Morphologies are defined in terms of two-point segments.
    * Gaps are allowed between segments anywhere in a morphology.
* Exposed the current `time` inside mechanisms.
* Added support for NeuroML2 morphology descriptions.
* Added a "stitch" morphology builder for constructing morphologies with
  cable sections that can connect to any location on their parent cable.                    
* Replaced recipe probe API with more flexible API that allows for sampling
  not only voltages at single locations, but currents, ion species properties,
  and mechanism state variables at single locations or across an entire cell.
* Added support for querying probe metadata from the simulation object.
* Added new 'place_pwlin' C++ API for cell geometry queries.
* Added support for loading Allen SDK cell model morphologies from SWC.
* Added support for composing policies for creating compartments over sub-regions.

### Documentation
* Restructured documentation to have cleaner separation between high level descriptions
  of concepts and the C++ and Python APIs.
* Added high level documentation for morphology descriptions, labels and cable cell
  construction.

### Optimizations
* Implemented memory optimizations for GPU matrix solver.
* Added support for ARM SVE intrinsics in the vectorized CPU back end.

### Bug Fixes
* Fixed various modcc code generation errors.



# v0.3

** 2020 04 01 **

**Arbor** library **version 0.3**, tagged as `v0.3`

Arbor is a library for implementing performance portable network simulations of multi-compartment neuron models.

An installation guide and library documentation are available online at [Read the Docs](https://arbor.readthedocs.io/en/latest/).

[Submit a ticket](https://github.com/eth-cscs/arbor) if you have any questions or want help.

## Change Log

Changes since [v0.2](https://github.com/arbor-sim/arbor/releases/tag/v0.2):

* Python wrapper with pip installation.
* Replace the morphology specification API for more flexible cell building.
* Flat descriptions of ion channel distribution and synapse placement.
* Multi-compartment back end support for sub-branch mechanism distributions.
* Improved NMODL support:
    * nonlinear kinetic schemes
    * linear system solution in initial conditions
    * many small features and bug fixes
* Generic ion species.
* Many optimizations and bug fixes.



# v0.2.1

** 2019 08 26 **

**Arbor** library **version 0.2.1**, tagged as `v0.2.1`

Arbor is a library for implementing performance portable network simulations of multi-compartment neuron models.

An installation guide and library documentation are available online at [Read the Docs](https://arbor.readthedocs.io/en/latest/).

[Submit a ticket](https://github.com/eth-cscs/arbor) if you have any questions or want help.

### Release Notes.

Minor Update.



# v0.2

** 2019 03 04 **

**Arbor** library **version 0.2**, tagged as `v0.2`

Arbor is a library for implementing performance portable network simulations of multi-compartment neuron models.

An installation guide and library documentation are available online at [Read the Docs](https://arbor.readthedocs.io/en/latest/).

[Submit a ticket](https://github.com/eth-cscs/arbor) if you have any questions or want help.


Some key features include:

  * Optimized back ends for CUDA, KNL, AVX2, ARM NEON intrinsics.
  * Asynchronous spike exchange that overlaps compute and communication.
  * Efficient sampling of voltage and current on all back ends.
  * Efficient implementation of all features on GPU.
  * Reporting of memory and energy consumption (when available on platform).
  * An API for addition of new cell types, e.g. LIF and Poisson spike generators.

## Change Log

Changes since [v0.1](https://github.com/arbor-sim/arbor/releases/tag/v0.1):

  * A new Hines matrix solver back end for the GPU that parallelises over cell
    branches, not cells, to increase the amount of parallelism. See #631.
  * Support for describing and simulating electrical gap junctions. See #661 #686.
  * An additional library `libarborenv` is now installed with useful
    helper functionality for managing the environment (e.g. detecting the number of available CPU cores). See #679.
  * Detection and allocation of GPUs to MPI ranks on systems with more than one GPU per node in `libarborenv`. See #659 and #654.
  * The miniapp example was removed and replaced with a simple single cell model that shows how to use morphologies. See #703 and #710.
  * Support for ARM NEON intrinsics. See #698.
  * Basic Python support. Full Python support is slated for v0.3. See #668.

### Contributors

Nora Abi Akar
John Biddiscombe
Benjamin Cumming
Felix Huber
Marko Kabic
Vasileios Karakasis
Wouter Klijn
Anne Küsters
Alexander Peyser
Stuart Yates

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2583709.svg)](https://doi.org/10.5281/zenodo.2583709)

Nora Abi Akar, John Biddiscombe, Benjamin Cumming, Felix Huber, Marko Kabic, Vasileios Karakasis, Wouter Klijn, Anne Küsters, Alexander Peyser, Stuart Yates. (2019, March 4). arbor-sim/arbor: Arbor Library v0.2 (Version v0.2). Zenodo. http://doi.org/10.5281/zenodo.2583709



# v0.1

** 2018 10 18 **

## Arbor Library

**Arbor** library **version 0.1**, tagged as `v0.1`

Arbor is a library for implementing performance portable network simulations of multi-compartment neuron models.

An installation guide and library documentation are available online at [Read the Docs](http://arbor.readthedocs.io/en/v0.1/).

[Submit a ticket](https://github.com/eth-cscs/arbor) if you have any questions or want help.


Some key features include:

    * Optimized back ends for CUDA, KNL and AVX2 intrinsics.
    * Asynchronous spike exchange that overlaps compute and communication.
    * Efficient sampling of voltage and current on all back ends.
    * Efficient implementation of all features on GPU.
    * Reporting of memory and energy consumption (when available on platform).
    * An API for addition of new cell types, e.g. LIF and Poisson spike generators.
    * Validation tests against numeric/analytic models and NEURON.

### Contributors
Nora Abi Akar
John Biddiscombe
Benjamin Cumming
Marko Kabic
Vasileios Karakasis 
Wouter Klijn
Anne Küsters
Ivan Martinez
Alexander Peyser
Stuart Yates

## Citation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1459679.svg)](https://doi.org/10.5281/zenodo.1459679)

If you use this version of Arbor, please cite it as **Nora Abi Akar, John Biddiscombe, Benjamin Cumming, Marko Kabic, Vasileios Karakasis, Wouter Klijn, Anne Küsters, Ivan Martinez, Alexander Peyser, Stuart Yates. (2018, October 12). arbor-sim/arbor: Version 0.1: First release (Version v0.1). Zenodo. http://doi.org/10.5281/zenodo.1459679**. The full citation is available in different formats on [Zenodo]( http://doi.org/10.5281/zenodo.1459679).
