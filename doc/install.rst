Installing Arbor
################

Installation of Arbor is done by checking out the source code and compiling it on
the target system.

This guide starts with an overview of the building process, and the various options
available to customize the build.
The guide then covers installation and running on HPC clusters, followed by a
troubleshooting guide for common build problems.

.. _install_requirements:

Requirements
============

Minimum Requirements
--------------------

The non distributed (i.e. no MPI) version of Arbor can be compiled on Linux or OS X systems
with very few tools.

.. table:: Required Tools

    =========== ============================================
    Tool        Notes
    =========== ============================================
    git         To check out the code, min version 2.0.
    cmake       To set up the build, min version 3.0.
    compiler    A C++11 compliant compiler. See `compilers <compilers_>`_.
    =========== ============================================


.. _compilers:

Compilers
~~~~~~~~~

Arbor requires a C++ compiler that fully supports C++11 (we have plans to move
to C++14 soon).
Arbor has been tested with gcc and clang extensively, and we are confident
that these two compilers will generate more efficient code than vendor-specific
compilers.

.. table:: Supported Compilers

    =========== ============ ============================================
    Compiler    Min version  Notes
    =========== ============ ============================================
    GCC         5.4.1        Earlier 5.x releases probably work.
    Clang       4.0          Clang 3.8 and later probably work.
    Apple Clang 9            Full support
    Intel       17           Full support
    =========== ============ ============================================

.. Note::
    The IBM xlc compiler versions 13.1.4 and 13.1.6 have been tested for compiling on IBM power 8.
    Arbor contains some patches to work around xlc compiler bugs, however we do not recommend using
    xlc because gcc produces much faster code, with much lower comilation times.

.. Note::
    Is is commonly assumed that to get the best performance one should use a vendor-specific
    compiler (e.g. the Intel, Cray or IBM compilers). These compilers are often better at
    generating auto-vectorized loops, however for everything else gcc and clang
    nearly always generate much more efficient code.

    The main computational loops in Arbor are cross compiled from NMODL, from which
    Arbor generates vectorized code, which can be compiled very efficiently by gcc
    and clang.
    This allows us to restrict our compilers to gcc and clang for faster compilation times, fewer
    compiler bugs to work around, and support for recent C++ standards.


Optional Requirements
---------------------

Python
~~~~~~

Python 3 is required if targetting the Python front end.

GPU Support
~~~~~~~~~~~

Arbor has full support for NVIDIA GPUs, for which the NVIDIA CUDA toolkit version 8 is required.

Distributed
~~~~~~~~~~~

To use cluster systems to run large scale simulations, Arbor uses MPI.
Arbor has been tested on MVAPICH2, OpenMPI, Cray MPI, and IBM MPI.
More information on building with MPI is in the HPC cluster section

Documentation
~~~~~~~~~~~~~~

To build a local copy of the html documentation that you are reading now, you will need to
install `Sphinx <http://www.sphinx-doc.org/en/master/>`_.

.. _downloading:

Getting the Code
================

The easiest way to acquire the latest version of Arbor is to check the code out from the `Github repository <https://github.com/eth-cscs/arbor>`_:

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

.. _building:

Building Arbor
==============

Before building an optimzed version for your target system, it is a good idea to build a debug version:

.. code-block:: bash

    # clone
    git clone https://github.com/eth-cscs/arbor.git --recursive
    cd arbor

    # make a path for building
    mkdir build
    cd build

    # use CMake to configure the build with default options
    cmake ..
    make -j 4

    # run tests
    ./test/test.exe
    ./test/global_communication.exe

This sequence of commands will build Arbor in debug mode with the default options.

Debugging
---------

Sometimes things go wrong: tests fail, simulations give strange results, segmentation
faults occur and exceptions are thrown.

A good first step when things to wrong is to turn on additional assertions that can
catch errors. These are turned off by default (because they slow things down a lot), and have to be turned on by setting
the ``ARB_WITH_ASSERTIONS`` CMake option:

.. code-block:: bash

    cmake -DARB_WITH_ASSERTIONS=ON

.. Note::
    These assertions are in the form of ``EXPECTS`` statements inside the code,
    for example:

    .. code-block:: cpp

        void decrement_min_remaining() {
            EXPECTS(min_remaining_steps_>0);
            if (!--min_remaining_steps_) {
                compute_min_remaining();
            }
        }

    A failing ``EXPECT`` statement indicates that an error inside the Arbor
    library, caused either by a logic error in Arbor, or incorrectly checked user input.

    If this occurs, it is highly recommended that you attach the output to the
    `bug report <https://github.com/eth-cscs/arbor/issues>`_ you send to the Arbor developers!


Optimization
------------

The default debug build target generates very slow code.
To use Arbor for interesting work, we should first compile it in release mode.

.. code-block:: bash

    cmake -DCMAKE_={debug, release}


Multi Threading
------------------

By default Arbor is built with the `cthread` threading back end, which is
part of Arbor itself.
This is fast enough for most applications, however there are some situations when
a more efficient threading implementation might be required:

* simulations with many small/light cells;
* running with many threads, such as on IBM Power 8 (80 threads/socket) or Intel
  KNL (>100 threads/socket).



.. _cluster:

HPC Clusters
============

HPC clusters offer their own unique challenges when compiling and running software,
so we cover some common issues in this section.
If you encounter unique challenges on your target system that are not covered here,
please make an issue on our `Github <https://github.com/eth-cscs/arbor/issues>`_.
We will do our best to help you directly, and update this guide to help future users.


Troubleshooting
===============

Issues to cover
* That annoying `CMP0023` cmake warning
* CMake warnings about missing git submodules
* Intel compiler uses gcc 4 headers
