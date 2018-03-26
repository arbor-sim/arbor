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

---

### `event_setup`

#### Motivation

Post synaptic events are generated by the communicator after it gathers the local spikes.

One set of events is generated for each cell group, in an unsorted `std::vector<post_synaptic_event>`.

Each cell group must take this unsorted vector, store the events, and for each integration interval generate a list events that are sorted first by target gid, then by delivery time.

As it is implemented, this step is a significant serialization bottleneck on the GPU back end, where one thread must process many events before copying them to the GPU.

This benchmark tries to understand the behavior of the current implementation, and test some alternatives.

#### Implementations

Three implementations are considered:

1. Single Queue (1Q) method (the current approach)
    1. All events to be delivered to a cell group are pushed into a heap
       based queue, ordered according to delivery time.
    2. To build the list of events to deliver before `tfinal`, events are
       popped off the queue until the head of the queue is an event to be
       delivered at or after `tfinal`. These events are `push_back`ed onto
       a `std::vector`.
    3. The event vector is `std::stable_sort`ed on target gid.

2. Multi Queue (NQ) method
    1. One queue is maintained for each cell in the cell group. The first
       phase pushes events into these smaller queues.
    2. The queues are visited one by one, and events before `tfinal` are
       `push_back` onto the single `std::vector`.

With this approach the events are partitioned by target gid for free, and the overheads of pushing and popping onto shorter queues should see speedup.

2. Multi Vector (NV) method
    1. A very similar approach to the NQ method, with a `std::vector`
       of events maintained for each cell instead of a priority queue.
    2. Events are `push_back`ed onto the vectors, which are then sorted
       and searched for the sub-range of events to be delivered in the next
       integration interval.

This approach has the same complexity as the NQ approach, but is a more "low-level" approach that uses `std::sort` to obtain, as opposed to the ad-hoc heap sort of popping from a queue.

#### Results

Platform:
* Xeon(R) CPU E5-2650 v4 (Haswell 12 cores @ 2.20 GHz)
* Linux 3.10.0
* gcc version 6.3.0

The benchmark varies the number of cells in the cell group, and the mean number of events per cell. The events are randomly generated in the interval `t in [0, 1]` and `target gid in {0, ..., ncells-1}`, with uniform distribution for both time and gid.

Below are benchmark results for 1024 events per cell as the number of cells varies.

For one cell there is little benefit with the NQ over 1Q, because in this case the only difference is avoiding the stable sort by gid.
The NV method is faster by over 2X for one cell, and the speedup increases to 7.8x for 10k cells.
Overall, maintaining seperate queues for each cell is much faster for more than one cell per cell group, and the additional optimizations of the NV method are significant enough to justifiy the more complicated implementation.

*time in ms*

|method|  1 cell  |   10 cells |  100 cells | 1k cells |  10k cells |
|------|----------|------------|------------|----------|------------|
|1Q    |   0.0597 | 1.139 | 18.74 | 305.90 | 5978.3 |
|nQ    |   0.0526 | 0.641 |  6.71 |  83.50 | 1113.1 |
|nV    |   0.0249 | 0.446 |  4.77 |  52.71 |  769.7 |

*speedup relative to 1Q method*

|method|  1 cell   |  10 cells |  100 cells | 1k cells |  10k cells |
|------|-----------|-----------|------------|----------|------------|
|1Q    |  1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
|nQ    |  1.1 | 1.8 | 2.8 | 3.7 | 5.4 |
|nV    |  2.4 | 2.6 | 3.9 | 5.8 | 7.8 |

---

### `default_construct`

#### Motivation

The `padded_allocator` code allows us to use, for example, a `std::vector` for CPU-side aligned storage and padded storage (for SIMD)
instead of the `memory::array` class. The latter though does not construct its elements, while a `std::vector` will use the allocator's
`construct` method.

For scalar values that have trivial default constructors, a `std::allocator` construction with no arguments will value-initialize,
which will zero initialize any non-class values. By supplying an alternate `construct` method, we can make an allocator that will
default-initialize instead, skipping any initialization for non-class values, and providing semantics similar to that of
`memory::array`.

Is it worth doing so?

#### Implementation

The microbenchmark uses an adaptor class that replaces the allocator `construct` methods to default initialize if there are no
arguments given. The benchmark creates a vector using the standard or adapted allocator, fills with the numbers from 1 to n
and takes the sum.

For comparison, the benchmark also compares the two vectors when they are initialized by a pair of iterators that provide the
same enumeration from 1 to n.

#### Results

With this low computation-to-size ratio task, using the default constructing adaptor gives a significant performance benefit.
With the iterator-pair construction however, where we would expect no performance difference, GCC (but not Clang) produces
very much slower code.

Note that Clang produces overall considerably faster code.

Platform:
* Xeon E3-1220 v2 with base clock 3.1 GHz and max clock 3.5 GHz. 
* Linux 4.9.75
* gcc version 7.3.1
* clang version 6.0.0
* optimization options: -O3 -march=ivybridge

##### Create then fill and sum

*GCC*

|    size  | value-initialized | default-initialized |
|---------:|------------------:|--------------------:|
|    1 kiB |            403 ns |              331 ns |
|    4 kiB |          1 430 ns |            1 142 ns |
|   32 kiB |         12 377 ns |            8 982 ns |
|  256 kiB |        114 598 ns |           81 599 ns |
| 1024 kiB |        455 502 ns |          323 366 ns |

*Clang*

|    size  | value-initialized | default-initialized |
|---------:|------------------:|--------------------:|
|    1 kib |            228 ns |              147 ns |
|    4 kib |            826 ns |              527 ns |
|   32 kib |         10 425 ns |            6 823 ns |
|  256 kib |        106 497 ns |           72 375 ns |
| 1024 kib |        430 561 ns |          293 999 ns |

##### Create directly from counting iterators and sum

*GCC*

|    size  | value-initialized | default-initialized |
|---------:|------------------:|--------------------:|
|    1 kiB |            335 ns |              775 ns |
|    4 kiB |          1 146 ns |            2 920 ns |
|   32 kiB |          8 954 ns |           23 197 ns |
|  256 kiB |         81 609 ns |          193 230 ns |
| 1024 kiB |        322 947 ns |          763 243 ns |

*Clang*

|    size  | value-initialized | default-initialized |
|---------:|------------------:|--------------------:|
|    1 kiB |            151 ns |              160 ns |
|    4 kiB |            531 ns |              528 ns |
|   32 kiB |          6 790 ns |            6 816 ns |
|  256 kiB |         72 460 ns |           72 687 ns |
| 1024 kiB |        293 991 ns |          293 746 ns |
