Installing
##############

Installation guide.

.. _install_requirements:

Before starting
===============

We will require:

  * git
  * cmake
  * C++11 compliant compiler

For GPU support:

  * NVIDIA CUDA toolkit

To make these docs you also need:

  * Sphinx

.. _downloading:

Downloading
======================================

The easiest way to acquire the latest version of Arbor is to check the code out from our GitHub repository:

.. code-block:: bash

    git clone https://github.com/eth-cscs/nestmc-proto.git

You can also point your browser to our `Github page <https://github.com/eth-cscs/nestmc-proto>`_ and download a zip file.

.. _install_desktop:

Basic Installation
======================================

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

