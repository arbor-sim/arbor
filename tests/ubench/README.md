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
sub-interval are given by a function of the index. This requires the computation of the sizes
> d<sub><i>i</i></sub> = Σ<sub><i>j</i>&lt;<i>i</i></sub> <i>f</i>(<i>j</i>).

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
*  gcc version 6.2.0
*  clang version 3.8.1

| Compiler    | direct/function | transform/function | direct/object | transform/object |
|:------------|----------------:|-------------------:|--------------:|-----------------:|
| g++ -O3     |  907 ns | 2090 ns |  907 ns | 614 ns |
| clang++ -O3 | 1063 ns |  533 ns | 1051 ns | 532 ns |

---

### `cuda_compare_and_reduce`

#### Motivation

One possible mechanism for determining if device-side event delivery had exhausted all
events is to see if the start and end of each event-span of each cell were equal or not.
This is equivalent to the test:

> ∃i: a[i] &lt; b[i]?

for device vectors _a_ and _b_.

How expensive is it simply to copy the vectors over to the host and compare there?
Benchmarking indicated that for vectors up to approximately 10^5 elements, it was
significiantly faster to copy to host, despite the limited host–device bandwidth.

This non-intuitive result appears to be the result of the high overhead of `cudaMalloc`;
pre-allocating space on the device for the reduction result restores expected
performance behaviour.

#### Implementations

Four implementations are considered:

1. Copy both vectors to host, run short-circuit compare.

2. Use `thrust::zip_iterator` and `thrust::any_of` to run the reduction on the device.

3. Use a short custom cuda kernel to run the reduction, using `__syncthreads_or` and
   a non-atomic write to the result.

4. Use the same cuda kernel as above, but pre-allocate the device memory to store the
   result.

Note: a fairer host-based comparison would interleave comparison with data transfer.

#### Results

Results here are presented for vector size _n_ equal to 256, 512, 32768 and 262144,
with the two vectors equal.

Platform:
* Xeon(R) CPU E5-2650 v4 with base clock 2.20 GHz and max clock 2.9 GHz.
* Tesla P100-PCIE-16GB
* Linux 3.10.0
* gcc version 5.3.0
* nvcc version 8.0.61

| _n_ | host copy | thrust | custom cuda | custom cuda noalloc |
|:----|----------:|-------:|------------:|--------------------:|
| 256    |  18265 ns |  41221 ns |  23255 ns | 16440 ns |
| 512    |  18466 ns | 286331 ns | 265113 ns | 16335 ns |
| 32768  | 102880 ns | 296836 ns | 265169 ns | 16758 ns |
| 262144 | 661724 ns | 305209 ns | 269095 ns | 19792 ns |

---

### `cuda_reduce_by_index`

#### Motivation

The reduction by key pattern with repeated keys is used when "point process"
mechanism contributions to currents are collected. More than one point process,
typically synapses, can be attached to a compartment, and when their
contributions are computed and added to the per-compartment current in
parallel, care must be taken to avoid race conditions. Early versions of Arbor
used cuda atomic operations to perform the accumulation, which works quite well
up to a certain point. However, performance with atomics decreases as the
number of synapses per compartment increases, i.e. as the number of threads
performing simultatneous atomic updates on the same location increases.

#### Implementations

Two implementations are considered:

1. Perform reductions inside each warp, which is a multi-step process:
    1. threads inside each warp determine which other threads they must perform a reduction with
    2. threads perform a binary reduction tree operation using warp shuffle intrinsics
    3. one thread performs a CUDA atomic update for each key.
    4. note that this approach takes advantage of the keys being sorted in ascending order

2. The naive (however simple) use of CUDA atomics.

#### Results

Platform:
* Xeon(R) CPU E5-2650 v4 (Haswell 12 cores @ 2.20 GHz)
* Tesla P100-PCIE-16GB
* Linux 3.10.0
* gcc version 5.2.0
* nvcc version 8.0.61

Results are presented as speedup for warp intrinsics vs atomics, for both
single and double precision. Note that the P100 GPU has hardware support for
double precision atomics, and we expect much larger speedup for double
precision on Keplar GPUs that emulate double precision atomics with CAS. The
benchmark updates `n` locations, each with an average density of `d` keys per
location. This is equivalent to `n` compartments with `d` synapses per
compartment. Atomics are faster for the case where both `n` and `d` are small,
however the gpu is backend is for throughput simulations, with large cell
groups with at least 10k compartments in total.

*float*

| _n_    | d=1  | d=10 | d=100| d=1000|
|--------|------|------|------|-------|
| 100    | 0.75 | 0.80 | 1.66 | 10.7  |
| 1k     | 0.76 | 0.87 | 3.15 | 12.5  |
| 10k    | 0.87 | 1.14 | 3.52 | 14.0  |
| 100k   | 0.92 | 1.34 | 3.58 | 15.5  |
| 1000k  | 1.18 | 1.43 | 3.53 | 15.2  |

*double*

| _n_    | d=1  | d=10 | d=100| d=1000|
|--------|------|------|------|-------|
| 100    | 0.91 | 0.94 | 1.82 |  9.0  |
| 1k     | 0.89 | 0.99 | 2.38 | 10.0  |
| 10k    | 0.94 | 1.09 | 2.42 | 11.1  |
| 100k   | 0.98 | 1.59 | 2.36 | 11.4  |
| 1000k  | 1.13 | 1.63 | 2.36 | 11.4  |
