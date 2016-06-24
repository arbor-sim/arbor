# cell_algorithms

```bash
# clone repo
git clone git@github.com:eth-cscs/cell_algorithms.git
cd cell_algorithms/

# setup sub modules
git submodule init
git submodule update

# setup environment
# on a desktop system this might not be required
# on a cluster this could be required
module load gcc
module load cmake
export CC=`which gcc`
export CXX=`which g++`

# build main project (out-of-tree)
mkdir build
cd build
cmake ..
make -j

# test
cd tests
./test.exe
```

## MPI

Set the `WITH_MPI` option either via the ccmake interface, or via the command line as shown below.
For the time being our CMake configuration does not try to detect MPI.
Instead, it relies on the user specifying an MPI wrapper for the compiler by setting the `CXX` and `CC` environment variables.

```
export CXX=mpicxx
export CC=mpicc
cmake <path to CMakeLists.txt> -DWITH_MPI=ON

```

## TBB

Support for multi-threading requires Intel Threading Building Blocks (TBB).
When TBB is installed, it comes with some scripts that can be run to set up the user environment.
The scripts set the `TBB_ROOT` environment variable, which is used by the CMake configuration to find TBB.


```
source <path to TBB installation>/tbbvars.sh
cmake <path to CMakeLists.txt> -DWITH_TBB=ON
```

### TBB on Cray systems

To compile with TBB on Cray systems, load the intel module, which will automatically configure the environment.

```
# load the gnu environment for compiling the application
module load PrgEnv-gnu
# gcc 5.x does not work with the version of TBB installed on Cray
# requires at least version 4.4 of TBB
module swap gcc/4.9.3
# load the intel programming module
# on Cray systems this automatically sets `TBB_ROOT` environment variable
module load intel
module load cmake
export CXX=`which CC`
export CC=`which cc`

# multithreading only
cmake <path to CMakeLists.txt> -DWITH_TBB=ON -DSYSTEM_CRAY=ON

# multithreading and MPI
cmake <path to CMakeLists.txt> -DWITH_TBB=ON -DWITH_MPI=ON -DSYSTEM_CRAY=ON

```
