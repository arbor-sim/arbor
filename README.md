# NestMC Prototype

This is the repository for the NestMC prototype code. Unfortunately we do not have thorough documentation of how-to guides.
Below are some guides for how to build the project and run the miniapp.
Contact us or submit a ticket if you have any questions or want help.
https://github.com/eth-cscs/nestmc-proto

1. Basic installation
2. MPI
3. TBB
4. TBB on Cray systems
5. Targeting KNL
6. Examples of environment configuration
    - Julia
    
## Basic installation
```bash
# clone repository
git clone git@github.com:eth-cscs/nestmc-proto.git
cd nestmc-proto/

# setup environment
# on a desktop system this is probably not required
# on a cluster this is usually required to make sure that an appropriate
# compiler is chosen.
module load gcc
module load cmake
export CC=`which gcc`
export CXX=`which g++`

# build main project (out-of-tree)
mkdir build
cd build
cmake <path to CMakeLists.txt>
make -j

# test
cd tests
./test.exe
```

## MPI

Set the `NMC_WITH_MPI` option either via the ccmake interface, or via the command line as shown below.
To ensure that CMake detects MPI correctly, you should specify the MPI wrapper for the compiler by setting the `CXX` and `CC` environment variables.

```
export CXX=mpicxx
export CC=mpicc
cmake <path to CMakeLists.txt> -DNMC_WITH_MPI=ON
```

## TBB

Support for multi-threading requires Intel Threading Building Blocks (TBB).
When TBB is installed, it comes with some scripts that can be run to set up the user environment.
The scripts set the `TBB_ROOT` environment variable, which is used by the CMake configuration to find TBB.

```
cmake <path to CMakeLists.txt> -DNMC_THREADING_MODEL=tbb
```

### TBB on Cray systems

TBB requires dynamic linking, which is not enabled by default in the Cray programming environment.
CMake is quite brittle, so take care to follow these step closely.
TBB provides a CMake package that will attempt to automatically download and compile TBB from within CMake.
Set the environment variable `CRAYPE_LINK_TYPE=dynamic`, to instruct the Cray PE linker to enable dynamic linking.
CMake (at least since CMake 3.6) will automatically detect the Cray programming environment, and will by default use static linking, unless the `CRAYPE_LINK_TYPE` environment variable has been set to `dynamic`.
Note, the CMake package provided by TBB is very fragile, and won't work if CMake is forced to use the `CrayLinuxEnvironment` as shown in the code below. Instead, let Cmake automatically detect the programming environment.

```
export CRAYPE_LINK_TYPE=dynamic
cmake <path-to-arbor-source> -DNMC_THREADING_MODEL=tbb

# NOTE: specifying CMAKE_SYSTEM_NAME won't work, instead let CMake automatically
# detect the build environment as above.
cmake <path-to-arbor-source> -DNMC_THREADING_MODEL=tbb  -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment
```

```
export CRAYPE_LINK_TYPE=dynamic
```

## targeting KNL

#### build modparser without KNL environment

The source to source compiler "modparser" that generates the C++/CUDA kernels for the ion channels and synapses is in a separate repository.
By default it will be built with the same compiler and flags that are used to build the miniapp and tests.

This can cause problems if we are cross compiling, e.g. for KNL, because the modparser compiler might not be runnable on the compilation node.
You are probably best of building the software twice: Once without KNL support to create the modcc parser and next the KNL version using
the now compiled executable

Modparser requires a C++11 compiler, and has been tested on GCC, Intel, and Clang compilers
  - if the default compiler on your is some ancient version of gcc you might need to load a module/set the CC and CXX environment variables.

CMake will look for the source to source compiler executable, `modcc`, in the `PATH` environment variable, and will use the version if finds instead of building its own.
So add the g++ compiled modcc to your path
e.g:

```bash
# First build a 'normal' non KNL version of the software

# Load your environment (see section 6 for detailed example)
export CC=`which gcc`; export CXX=`which g++`

# Make directory , do the configuration and build
mkdir build
cd build
cmake <path to CMakeLists.txt> -DCMAKE_BUILD_TYPE=release
make -j8

# set path and test that you can see modcc
export PATH=`pwd`/modcc:$PATH
which modcc
```

#### set up environment

- source the intel compilers
- source the TBB vars
- I have only tested with the latest stable version from on-line, not the version that comes installed sometimes with the Intel compilers.

#### build miniapp

```bash
# clone the repository and set up the submodules
git clone https://github.com/eth-cscs/nestmc-proto.git
cd nestmc-proto

# make a path for out of source build
mkdir build_knl
cd build_knl

# run cmake with all the magic flags
export CC=`which icc`
export CXX=`which icpc`
cmake <path to CMakeLists.txt> -DCMAKE_BUILD_TYPE=release -DNMC_THREADING_MODEL=tbb -DNMC_WITH_PROFILING=ON -DNMC_VECTORIZE_TARGET=KNL
make -j
```

The flags passed into cmake are described:
  - `-DCMAKE_BUILD_TYPE=release` : build in release mode with `-O3`.
  - `-DNMC_THREADING_MODEL=tbb` : use TBB for threading on multi-core
  - `-DNMC_WITH_PROFILING=ON` : use internal profilers that print profiling report at end
  - `-DNMC_VECTORIZE_TARGET=KNL` : generate AVX512 instructions, alternatively you can use:
    - `AVX2` for Haswell & Broadwell
    - `AVX` for Sandy Bridge and Ivy Bridge

Currently, the Intel compiler is required when you specify a vectorize target.

#### run tests

Run some unit tests
```bash
cd tests
./test.exe
cd ..
```

## run miniapp

The miniapp is the target for benchmarking.
First, we can run a small problem to check the build.
For the small test run, the parameters have the following meaning
  - `-n 1000` : 1000 cells
  - `-s 200` : 200 synapses per cell
  - `-t 20`  : simulated for 20ms
  - `-p 0`   : no file output of voltage traces

The number of cells is the number of discrete tasks that are distributed to the threads in each large time integration period.
The number of synapses per cell is the amount of computational work per cell/task.
Realistic cells have anywhere in the range of 1,000-10,000 synapses per cell.

```bash
cd miniapp

# a small run to check that everything works
./miniapp.exe -n 1000 -s 200 -t 20 -p 0

# a larger run for generating meaningful benchmarks
./miniapp.exe -n 2000 -s 2000 -t 100 -p 0
```

This generates the following profiler output (some reformatting to make the table work):

```
              ---------------------------------------
             |       small       |       large       |
-------------|-------------------|-------------------|
total        |  0.791     100.0  | 38.593     100.0  |
  stepping   |  0.738      93.3  | 36.978      95.8  |
    matrix   |  0.406      51.3  |  6.034      15.6  |
      solve  |  0.308      38.9  |  4.534      11.7  |
      setup  |  0.082      10.4  |  1.260       3.3  |
      other  |  0.016       2.0  |  0.240       0.6  |
    state    |  0.194      24.5  | 23.235      60.2  |
      expsyn |  0.158      20.0  | 22.679      58.8  |
      hh     |  0.014       1.7  |  0.215       0.6  |
      pas    |  0.003       0.4  |  0.053       0.1  |
      other  |  0.019       2.4  |  0.287       0.7  |
    current  |  0.107      13.5  |  7.106      18.4  |
      expsyn |  0.047       5.9  |  6.118      15.9  |
      pas    |  0.028       3.5  |  0.476       1.2  |
      hh     |  0.006       0.7  |  0.096       0.2  |
      other  |  0.026       3.3  |  0.415       1.1  |
    events   |  0.005       0.6  |  0.125       0.3  |
    sampling |  0.003       0.4  |  0.051       0.1  |
    other    |  0.024       3.0  |  0.428       1.1  |
  other      |  0.053       6.7  |  1.614       4.2  |
-----------------------------------------------------
```

## Examples of environment configuration
### Julia (HBP PCP system)
``` bash
module load cmake
module load intel-ics
module load openmpi_ics/2.0.0
module load gcc/6.1.0
```

