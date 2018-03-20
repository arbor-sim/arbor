Installing Arbor
################

Installation of Arbor is done by checking out the source code and compiling it on
the target system.

This guide starts with an overview of the building process, and the various options
available to customize the build.
The guide then covers installation and running on `HPC clusters <_cluster>`_, followed by a
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
We recommend using GCC or Clang, for which Arbor has been tested and optimised.

.. table:: Supported Compilers

    =========== ============ ============================================
    Compiler    Min version  Notes
    =========== ============ ============================================
    GCC         5.2.0        Earlier 5.x releases probably work.
    Clang       4.0          Clang 3.8 and later probably work.
    Apple Clang 9
    Intel       17.0.1       Needs GCC 5 or later for std library.
    =========== ============ ============================================

.. Note::
    The ``CC`` and ``CXX`` environment variables are used to specify the compiler executable
    to the CMake build scripts. If these are not set, CMake may use a different compiler.
    On the system that the test below was performed, CMake used ``/usr/bin/c++``,
    which was GCC version 4.8.5.

    .. code-block:: bash

        # check which version of gcc is available
        $ g++ --version
        g++ (GCC) 5.2.0
        Copyright (C) 2015 Free Software Foundation, Inc.

        # set environment variables for compilers
        $ export CC=`which gcc`; export CXX=`which g++`;

        # launch CMake
        # the compiler version and path is given in the CMake output
        $ cmake ..
        -- The C compiler identification is GNU 5.2.0
        -- The CXX compiler identification is GNU 5.2.0
        -- Check for working C compiler: /cm/local/apps/gcc/5.2.0/bin/gcc
        -- Check for working C compiler: /cm/local/apps/gcc/5.2.0/bin/gcc -- works
        ...

.. Note::
    Is is commonly assumed that to get the best performance one should use a vendor-specific
    compiler (e.g. the Intel, Cray or IBM compilers). These compilers are often better at
    auto-vectorizing loops, however for everything else GCC and Clang nearly always generate
    much more efficient code.

    The main computational loops in Arbor are cross compiled from NMODL, from which
    Arbor generates vectorized code, which can be compiled very efficiently by GCC
    and Clang.
    This allows us focus on GCC and Clang for faster compilation times, fewer
    compiler bugs to work around, and support for recent C++ standards.

.. Note::
    The IBM xlc compiler versions 13.1.4 and 13.1.6 have been tested for compiling on
    IBM power 8. Arbor contains some patches to work around xlc compiler bugs,
    however we do not recommend using xlc because GCC produces much faster code,
    with much lower comilation times.

Optional Requirements
---------------------

GPU Support
~~~~~~~~~~~

Arbor has full support for NVIDIA GPUs, for which the NVIDIA CUDA toolkit version 8 is required.

Distributed
~~~~~~~~~~~

Arbor uses MPI to run on HPC cluster systems.
Arbor has been tested on MVAPICH2, OpenMPI, Cray MPI, and IBM MPI.
More information on building with MPI is in the `HPC cluster section <cluster_>`_.

Documentation
~~~~~~~~~~~~~~

To build a local copy of the html documentation that you are reading now, you will need to
install `Sphinx <http://www.sphinx-doc.org/en/master/>`_.

.. _downloading:

Getting the Code
================

The easiest way to acquire the latest version of Arbor is to check the code out from
the `Github repository <https://github.com/eth-cscs/arbor>`_:

.. code-block:: bash

    git clone https://github.com/eth-cscs/arbor.git --recursive

We recommend using a recursive checkout, because Arbor uses git submodules for some
of its library dependencies.
The CMake configuration attempts to detect if a required submodule is available, and
will print a helpful warning
or error message if not, but it is up to the user to ensure that all required
submodules are downloaded.

The git submodules can be updated, or initialized in a project that didn't use a
recursive checkout:

.. code-block:: bash

    git submodule update --init --recursive

You can also point your browser to Arbor's
`Github page <https://github.com/eth-cscs/arbor>`_ and download a zip file.
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

Quick Start: Examples
---------------------

Below are some example of CMake configurations for Arbor. For more detail on individual CMake parameters and flags, follow links to the more detailed descriptions below.

.. topic:: `Debug <buildtarget_>`_ mode with `assertions <debugging_>`_, `single threaded <threading_>`_ with `Clang <compilers_>`_.

    .. code-block:: bash

        export CC=`which clang`
        export CXX=`which clang++`
        cmake .. -DARB_WITH_ASSERTIONS=ON -DARB_THREADING_MODEL=serial

.. topic:: `Release <buildtarget_>`_ mode (i.e. build with optimization flags)

    .. code-block:: bash

        cmake .. -DCMAKE_BUILD_TYPE=release

.. topic:: `Release <buildtarget_>`_ mode on `Haswell <vectorize_>`_ with `cthread threading <threading_>`_

    .. code-block:: bash

        cmake .. -DCMAKE_BUILD_TYPE=release -DARB_THREADING_MODEL=cthread -DARB_VECTORIZE_TARGET=AVX2

.. topic:: `Release <buildtarget_>`_ mode on `KNL <vectorize_>`_ with `TBB threading <threading_>`_

    .. code-block:: bash

        cmake .. -DCMAKE_BUILD_TYPE=release -DARB_THREADING_MODEL=tbb -DARB_VECTORIZE_TARGET=KNL

.. topic:: `Release <buildtarget_>`_ mode with `CUDA <gpu_>`_ and `AVX2 <vectorize_>`_ and `GCC 5 <compilers_>`_

    .. code-block:: bash

        export CC=gcc-5
        export CXX=g++-5
        cmake .. -DCMAKE_BUILD_TYPE=release -DARB_VECTORIZE_TARGET=AVX2 -DARB_WITH_CUDA=ON



.. _buildtarget:

Build Target
------------

By default, Arbor is built in debug mode, which is very slow.
Arbor should be built in `release` mode, by setting the standard CMake
``CMAKE_BUILD_TYPE`` parameter.

.. code-block:: bash

    cmake -DCMAKE_BUILD_TYPE={debug,release}

..  _vectorize:

Vectorization
-------------

Explicit vectorization of key computational kernels can be enabled in Arbor by setting the
``ARB_VECTORIZE_TARGET`` CMake parameter:

.. code-block:: bash

    cmake -DARB_VECTORIZE_TARGET={none,KNL,AVX2,AVX512}

By default the ``none`` target is selected, which relies on compiler auto-vectorization.

    cmake -DARB_VECTORIZE_TARGET={none,KNL,AVX2,AVX512}

.. Warning::
    The vectorization target must be supported by the target architecture.
    A sure sign that an unsuported vectorization was chosen is an ``Illegal instruction``
    error at runtime. In the example below, we attempt to run the unit tests for an ``AVX2``
    build on an Ivy Bridge CPU, which does not support them

    .. code-block:: none

        $ ./tests/test.exe
        [==========] Running 581 tests from 105 test cases.
        [----------] Global test environment set-up.
        [----------] 15 tests from algorithms
        [ RUN      ] algorithms.parallel_sort
        Illegal instruction

    See the hints on `cross compiling <crosscompiling_>`_ if you get illegal instruction
    errors when trying to cross compile.

.. Note::
    The vectorization selection will change soon, to an interface with two parameters. The first
    will toggle vectorization, and the second will specify a specific architecture to target.
    For example, to generate optimized code for Intel Broadwell (i.e. AVX2 intrinsics):

    .. code-block:: bash

        cmake -DCMAKE_BUILD_TYPE=release -DARB_VECTORIZE=ON -DARB_ARCH=broadwell


.. _threading:

Multi Threading
---------------

Arbor provides three threading back ends, one of which is selected at compile time.
by setting the ``ARB_THREADING_MODEL`` CMake option:

.. code-block:: bash

    cmake -DARB_THREADING_MODEL={serial,cthread,tbb}

By default Arbor is built with multithreading enabled, using the `cthread`,
which is implemented in the Arbor source code.


.. table:: Threading Models.

    =========== ============== =================================================
    Model       Source         Description
    =========== ============== =================================================
    cthread         Arbor      Default. Multithreaded, based on C++11 ``std::thread``.
    serial          Arbor      Single threaded.
    tbb         git submodule  `Intel TBB <https://www.threadingbuildingblocks.org/>`_.
                               Recommended when using many threads.
    =========== ============== =================================================

.. Note::
    The default `cthread` threading is suitable for most applications.
    However there are some situations when the overheads of the threading runtime
    become significant. This is often the case for:

    * simulations with many small/light cells (e.g. LIF cells);
    * running with many threads, such as on IBM Power 8 (80 threads/socket) or Intel
      KNL (64-256 threads/socket).

    The TBB threading back end is highly optimized, and well suited to these cases.


.. Note::
    If the TBB back end is selected, Arbor's CMake uses a git submodule of the TBB
    repository to build and link a static version of the the TBB library. If you get
    an error stating that the TBB submodule is not available, you must update the git
    submodules:

    .. code-block:: bash

        git submodule update --init --recursive

.. Note::
    The TBB back end can be used on IBM Power 8 systems.

.. _gpu:

GPU Backend
-----------

Arbor supports NVIDIA GPUs using CUDA. The CUDA back end is enabled by setting the CMake ``ARB_WITH_CUDA`` option.

.. code-block:: bash

    cmake .. -DARB_WITH_CUDA=ON

.. Note::
    Abor requires CUDA version >= 8, and targets P100 GPUs.

.. _cluster:

HPC Clusters
============

HPC clusters offer their own unique challenges when compiling and running software,
so we cover some common issues in this section.
If you encounter unique challenges on your target system that are not covered here,
please make an issue on our `Github <https://github.com/eth-cscs/arbor/issues>`_.
We will do our best to help you directly, and update this guide to help future users.

MPI
---

Arbor uses MPI for distributed systems. By default it is built without MPI support, which
can enabled by setting the ``DARB_DISTRIBUTED_MODEL`` CMake parameter.
An example of building Arbor with MPI, high-performance threading and optimizations enabled
is:

.. code-block:: bash

    # set the compiler wrappers
    export CC=`which mpicc`
    export CXX=`which mpicxx`

    # configure with mpi, tbb threading and compiled with optimizations
    cmake .. -DARB_DISTRIBUTED_MODEL=mpi \
             -DCMAKE_BUILD_TYPE=release  \
             -DARB_THREADING_MODEL=tbb   \

    # run unit tests for global communication on 2 MPI ranks
    mpirun -n 2 ./tests/global_communication.exe

The first step to building with MPI support is to set the ``CC`` and ``CXX`` environment variables to refer to the mpi compiler wrappers.

.. Note::
    MPI distributions provide *compiler wrappers* for compiling MPI applications.

    In the example above the C and C++ compiler wrappers for C and C++ called
    ``mpicc`` and ``mpicxx`` respectively. The name of the compiler wrapper
    is dependent on the MPI distribution.

    The wrapper forwards the compilation to a compiler, like GCC, and
    you have to ensure that this compiler is able to compile Arbor. For wrappers
    that call GCC, Intel or Clang compilers, you can pass the ``--version`` flag
    to the wrapper. For example, on a Cray system where the C++ wrapper is called ``CC``:

    .. code-block:: bash

        $ CC --version
        g++ (GCC) 6.2.0 20160822 (Cray Inc.)

Cray Systems
------------

The compiler used by the MPI wrappers is set using a "programming enviroment" module.
The first thing to do is change this module, which by default is set to the Cray
programming environment.
For example, to use the GCC compilers, select the GNU programming enviroment:

.. code-block:: bash

    module swap PrgEnv-cray PrgEnv-gnu

The version of the compiler can then be set by choosing an appropriate gcc module.
In the example below we use ``module avail`` to see which versions of GCC are available,
then choose GCC 7.1.0

.. code-block:: bash

    $ module avail gcc      # see all available gcc versions

    ------------------------- /opt/modulefiles ---------------------------
    gcc/4.9.3    gcc/6.1.0    gcc/7.1.0    gcc/5.3.0(default)    gcc/6.2.0

    $ module swap gcc/7.1.0 # swap gcc 5.3.0 for 7.1.0

    $ CC --version          # test that the wrapper uses gcc 7.1.0
    g++ (GCC) 7.1.0 20170502 (Cray Inc.)

    # set compiler wrappers
    $ export CC=`which cc`; export CXX=`which CC`;

Note that the C and C++ compiler wrappers are, rather confusingly, called
``cc`` and ``CC`` respectively on Cray systems.

CMake detects that it is being run in the Cray programming environment, which makes
our lives a little bit more difficult (CMake sometimes tries a bit too hard to help).
To get CMake to correctly link our code, we need to set the ``CRAYPE_LINK_TYPE``
enviroment variable to ``dynamic``.

.. code-block:: bash

    export CRAYPE_LINK_TYPE=dynamic

Putting it all together, a typicaly workflow to configure the environment and CMake,
then build Arbor is:

.. code-block:: bash

    export CRAYPE_LINK_TYPE=dynamic
    module swap PrgEnv-cray PrgEnv-gnu
    moudle swap gcc/7.1.0
    export CC=`which cc`; export CXX=`which CC`;
    cmake .. -DARB_DISTRIBUTED_MODEL=mpi \      # MPI support
             -DCMAKE_BUILD_TYPE=release  \      # optimized
             -DARB_THREADING_MODEL=tbb   \      # tbb threading
             -DARB_SYSTEM_TYPE=Cray             # turn on Cray specific options

.. Note::
    If ``CRAYPE_LINK_TYPE`` isn't set, there will be warnings like the following when linking:

    .. code-block:: none

        warning: Using 'dlopen' in statically linked applications requires at runtime
                 the shared libraries from the glibc version used for linking

    Often the library or executable will work, however if a different glibc is loaded,
    Arbor will crash at runtime with obscure errors that are very difficult to debug.


Troubleshooting
===============

.. _crosscompiling:

Cross Compiling NMODL
---------------------

Care must be taken when Arbor is compiled on a system with a different architecture to the target system where Arbor will run.

This occurs quite frequently on HPC systems, for example when building on a login/service node that has a different architecture to the compute nodes.

Here we will use the example of compiling for Intel KNL, which typically has Intel Xeon CPUs on login nodes that don't support the AVX512 instructions used by KNL.

.. _debugging:

Debugging
---------

Sometimes things go wrong: tests fail, simulations give strange results, segmentation
faults occur and exceptions are thrown.

A good first step when things to wrong is to turn on additional assertions that can
catch errors. These are turned off by default (because they slow things down a lot),
and have to be turned on by setting the ``ARB_WITH_ASSERTIONS`` CMake option:

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



Other...
--------

Issues to cover:
    * That annoying `CMP0023` cmake warning
    * CMake warnings about missing git submodules
    * Intel compiler uses GCC 4 headers

