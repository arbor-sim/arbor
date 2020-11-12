# C++ Examples

A series of mini-apps to demonstrate specific features.
Please refer to the respective sub-directories for more
information.

All examples can be built by setting up the standard
CMake build process, e.g.

    # go to arbor's source directory   
    cd arbor
    # set up a build dir
    mkdir build
    cd build
    # configure (example options for an MPI-enabled build)
    cmake .. -DARB_VECTORIZE=ON -DARB_WITH_MPI=ON -DCMAKE_CXX_COMPILER=`which g++`
    make examples
    
