# Library microbenchmarks

The benchmarks here are intended to:
* answer questions regarding choices of implementation in the library where performance is a concern;
* track the performance behaviour of isolated bits of library functionality across different platforms.


## Building and running

The micro-benchmarks are not built by default. After configuring CMake, they can be built with
`make ubenches`. Each benchmark is provided by a stand-alone C++ source file in `tests/ubench`;
the resulting executables are found in `test/ubench` relative to the build directory.

[Google benchmark](https://github.com/google/benchmark) is used as a harness. It is included
in the repository via a git submodule, and the provided CMake scripts will attempt to
run `git submodule update --init` on the submodule if it appears not to have been instantiated.


## Adding new benchmarks

New benchmarks are added by placing the corresponding implementation as a stand-alone
`.cpp` file in `tests/ubench` and adding the name of this file to the list `bench_sources`
in `tests/ubench/CMakeLists.txt`.

Each new benchmark should also have a corresponding entry in this `README.md`, describing
the motivation for the test and summarising at least one benchmark result.

Results in this file are destined to become out of date; we should consider some form
of semi-automated registration of results in a database should the number of benchmarks
become otherwise unwieldy.


## Benchmarks

### `accumulate_functor_values`

#### Motivation

The problem arises when constructing the partition of an integral range where the sizes of each
sub-interval are given by a function of the index. This requires the computation of
> d<sub><i>i</i></sub> = Σ<sub><i>j</i>&lt;<i>i</i> <i>f</i>(<i>j</i>).

One approach using the provided range utilities is to use `std::partial_sum` with
`util::transform_view` and `util::span`; the other is to simply write a loop that
performs the accumulation directly. What is the extra cost, if any, of the
transform-based approach?

The micro-benchmark compares the two implementations, where the function is a simple
integer square operation, called either via a function pointer or a functional object.

#### Results

Results here are presented only for vector size _n_ equal to 1024.

Platform:
*  Xeon E3-1220 v2 with base clock 3.1 GHz and max clock 3.5 GHz. 
*  Linux 4.4.34
*  gcc version 6.2.0 and clang version 3.8.1

| Compiler    | direct/function | transform/function | direct/object | transform/object |
|-------------|-----------------|--------------------|---------------|------------------|
| g++ -O3     | 907 ns | 2090 ns | 907 ns | 614 ns |
| clang++ -O3 | 1063 ns | 533 ns | 1051 ns | 532 ns |



