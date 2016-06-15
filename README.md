# cell_algorithms

```bash
# clone repo
git clone git@github.com:eth-cscs/cell_algorithms.git
cd cell_algorithms/

# setup sub modules
git submodule init
git submodule update

# setup environment
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
