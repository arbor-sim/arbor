.. _in_python:

Python Installation
===================

Arbor's Python API will be the most convenient interface for most users.

.. note::
    Arbor requires Python version 3.6 and later. It is advised that you update `pip` as well.
    We strongly encourage using ``pip`` to install Arbor.

Getting Arbor
-------------

Every point release of Arbor is pushed to the Python Package Index.
For x86-64 Linux and MacOS platforms, we provide binary wheels.
The easiest way to get Arbor is with
`pip <https://packaging.python.org/tutorials/installing-packages>`_:

.. code-block:: bash

    pip3 install arbor

.. note::
    For other platforms, `pip` will build Arbor from source.
    You will need to have some development packages installed in order to build Arbor this way.

    * Ubuntu/Debian: `git cmake gcc python3-dev python3-pip libxml2-dev`
    * Fedora/CentOS/OpenSuse: `git cmake gcc-c++ python3-devel python3-pip libxml2-devel`
    * MacOS: get `brew` `here <https://brew.sh>`_ and run `brew install cmake clang python3 libxml2`
    * Windows: the simplest way is to use `WSL <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ and then follow the instructions for Ubuntu.

To test that Arbor is available, try the following in a Python interpreter
to see information about the version and enabled features:

.. code-block:: python

    >>> import arbor
    >>> print(arbor.__version__)
    >>> print(arbor.__config__)

You are now ready to use Arbor! You can continue reading these documentation pages, have a look at the
:ref:`Python API reference<pyoverview>`, or visit the :ref:`tutorial`.

.. Note::
    To get help in case of problems installing with pip, run pip with the ``--verbose`` flag, and attach the output
    (along with the pip command itself) to a ticket on `Arbor's issues page <https://github.com/arbor-sim/arbor/issues>`_.

.. _in_python_custom:

Customising Arbor
^^^^^^^^^^^^^^^^^

If you wish to get the latest Arbor straight from
the master branch in our git repository, you can run:

.. code-block:: bash

    pip3 install git+https://github.com/arbor-sim/arbor.git

If you want to work on Arbor's code, you can get a copy of our repo and point `pip` at the local directory:

.. code-block:: bash

    # get your copy of the Arbor source
    git clone https://github.com/arbor-sim/arbor.git --recursive
    # make your changes and then instruct pip to build and install the local source
    pip3 install ./arbor/

Every time you make changes to the code, you'll have to repeat the second step.

Advanced options
^^^^^^^^^^^^^^^^

By default Arbor is installed with multi-threading enabled. To enable more
advanced forms of parallelism and other features, Arbor comes with a few
compilation options. These are of the form ``-D<KEY>=<VALUE>``, must be appended
to the ``pip`` invocation via ``--install-option='-D<...> -D<...> ...'`` and can
be used on both local (``pip3 install ./arbor``) and remote (``pip3 install
arbor``) copies of Arbor. See the examples below.

.. Note::

   ``pip`` compiles the Arbor C++ library and wrapper, as well as dependencies
    you might not have had installed yet (e.g. ``numpy``). It may take a few
    minutes. Pass the ``--verbose`` flag to pip to see the individual steps
    being performed if you are concerned that progress is halting.

    If you had Arbor installed already, you may need to remove it first before
    you can (re)compile it with the flags you need.

.. Warning::

   If you run into build issues while experimenting with build options, be sure
   to remove the ``_skbuild`` directory.

The following flags can be used to configure the installation:

* ``ARB_WITH_NEUROML=<ON|OFF>``: Enable support for NeuroML2 morphologies,
  requires ``libxml2`` library. Default ``OFF``
* ``ARB_WITH__MPI=<ON|OFF>``: Enable MPI support, requires MPI library.
  Default ``OFF``.
* ``ARB_GPU=<none|cuda|cuda-clang|hip>``: Enable GPU support for NVIDIA GPUs
  with nvcc using ``cuda``, or with clang using ``cuda-clang`` (both require
  cudaruntime). Enable GPU support for AMD GPUs with hipcc using ``hip``. By
  default set to ``none``, which disables GPU support.
* ``ARB_VECTORIZE=<ON|OFF>``: Enable vectorization. The architecture argument,
  documented below, may also have to be set appropriately to generated
  vectorized code. See :ref:`install-architecture` for details.
* ``ARB_ARCH=<native|*>``: CPU micro-architecture to target. The advised
  default is ``native``. See `here
  <https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html>`_ for a full list of
  options.

.. note::

   There are more, advanced flags that can be set. We are using ``sk-build`` and
   ``CMake`` under the hood, so all flags and options valid in ``CMake`` can be
   used in this fashion.

   Detailed instructions on how to install using CMake are in the :ref:`Python
   configuration <install-python>` section of the :ref:`installation guide
   <in_build_install>`. CMake is recommended if you need more control over
   compilation and installation, plan to use Arbor with C++, or if you are
   integrating with package managers such as Spack and EasyBuild.

In the examples below we assume you are installing from a local copy.

**Vanilla install** with no additional features enabled:

.. code-block:: bash

    pip3 install ./arbor

**With MPI support**. This might require loading an MPI module or setting the ``CC`` and ``CXX``
:ref:`environment variables <install-mpi>`:

.. code-block:: bash

    pip3 install ./arbor --install-option='-DARB_WITH_MPI=ON'

**Compile with** :ref:`vectorization <install-vectorize>` on a system with a SkyLake
:ref:`architecture <install-architecture>`:

.. code-block:: bash

    pip3 install ./arbor --install-option='-DARB_VECTORIZE=ON -DARB_ARCH=skylake'

**Enable NVIDIA GPUs (compiled with nvcc)**. This requires the :ref:`CUDA toolkit <install-gpu>`:

.. code-block:: bash

    pip3 install ./arbor --install-option='-DARB_GPU=cuda'

**Enable NVIDIA GPUs (compiled with clang)**. This also requires the :ref:`CUDA toolkit <install-gpu>`:

.. code-block:: bash

    pip3 install ./arbor --install-option='-DARB_GPU=cuda-clang'

**Enable AMD GPUs (compiled with hipcc)**. This requires setting the ``CC`` and ``CXX``
:ref:`environment variables <install-gpu>`

.. code-block:: bash

    pip3 install ./arbor --install-option='-DARB_GPU=hip'

Note on performance
-------------------

The Python interface can incur significant memory and runtime overheads relative to C++
during the *model building* phase, however simulation performance is the same
for both interfaces.
