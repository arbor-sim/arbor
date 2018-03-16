Installing
##############

Installation of Arbor is done by checking out the source code and compiling it on the target system.

This guide starts with an overview of the building process, and the various options available to
customize the build.
The guide then covers installation and running on HPC clusters, followed by a troubleshooting guide for
common build problems.

.. _install_requirements:

Before Starting
===============

To check the code out, or update submodules, git is required.

  * git

  * cmake 3.0

CMake

  * C++11 compliant compiler

For GPU support:

  * NVIDIA CUDA toolkit 8.0

To make these docs you also need:

  * Sphinx

.. _downloading:

Getting the Code
================

The easiest way to acquire the latest version of Arbor is to check the code out from the GitHub repository:

.. code-block:: bash

    git clone https://github.com/eth-cscs/arbor.git --recursive

We recommend using a recursive checkout, because Arbor uses git submodules for some of its library dependencies.
The CMake configuration attempts to detect if a required submodule is available, and will print a helpful warning
or error message if not, but it is up to the user to ensure that all required submodules are downloaded.

The git submodules can be updated, or initialized in a project that didn't use a recursive checkout:

.. code-block:: bash

    git submodule update --init --recursive

You can also point your browser to Arbor's `Github page <https://github.com/eth-cscs/arbor>`_ and download a zip file.
If you use the zip file, then don't forget to run git submodule update manually.

.. _install:

Installation
============

Before building an optimzed version for your target system, it is a good idea to build a debug version:

.. code-block:: bash

    # make a path for building
    mkdir build
    cd build

    # configure and build
    cmake ..
    make -j 4

    # run tests
    ./test/test.exe
    ./test/global_communication.exe

HPC Clusters
============

HPC clusters offer their own unique challenges when compiling and runing software,
so we cover some common issues in this seciont.
If you encounter unique challenges on your target system that are not covered here,
please make an issue on our `Github <https://github.com/eth-cscs/arbor/issues>`_.
We will do our best to help you directly, and update this guide to help future users.

Troubleshooting
===============

Issues to cover
* That annoying `CMP0023` cmake warning
* CMake warnings about missing git submodules
