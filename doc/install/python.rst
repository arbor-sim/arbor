.. _in_python:

Python Installation
===================

Arbor's Python API will be the most convenient interface for most users.

.. note::
    Arbor requires Python version 3.7 and later. It is advised that you update ``pip`` as well.
    We strongly encourage using ``pip`` to install Arbor.
    
    To get help in case of problems installing with pip, run pip with the ``--verbose`` flag, and attach the output
    (along with the pip command itself) to a ticket on `Arbor's issues page <https://github.com/arbor-sim/arbor/issues>`_.

Getting Arbor
-------------

Every point release of Arbor is pushed to the Python Package Index.
For x86-64 Linux and MacOS platforms, we provide binary wheels.
The easiest way to get Arbor is with
`pip <https://packaging.python.org/tutorials/installing-packages>`_:

.. code-block:: bash

    pip3 install arbor

To test that Arbor is available, try the following in a Python interpreter
to see information about the version and enabled features:

.. code-block:: python

    >>> import arbor
    >>> print(arbor.__version__)
    >>> print(arbor.__config__)

You are now ready to use Arbor! You can continue reading these documentation pages, have a look at the
:ref:`Python API reference<pyoverview>`, or visit the :ref:`tutorial`.

.. Warning::
    
    For builds from Arbor's source, you will need to have some development packages installed. Installing Arbor
    for any other platforms than listed above, ``pip`` will attempt a build from source and thus require these
    packages as well.

    * Ubuntu/Debian: ``git cmake gcc python3-dev python3-pip``
    * Fedora/CentOS/OpenSuse: ``git cmake gcc-c++ python3-devel python3-pip``
    * MacOS: get ``brew`` `here <https://brew.sh>`_ and run ``brew install cmake clang python3``
    * Windows: the simplest way is to use `WSL <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ and then follow the instructions for Ubuntu.

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

.. _in_python_adv:

Advanced options
^^^^^^^^^^^^^^^^

Arbor comes with a few compilation options, some of them related to advanced forms of parallelism and other features.
The options and flags are the same :ref:`as documented for the CMAKE build <quickstart>`, but they are passed differently.
To enable more, they must be placed in the ``CMAKE_ARGS`` environment variable.
The simplest way to do this is by prepending the ``pip`` command with ``CMAKE_ARGS=""``,
where you place the arguments separated by space inside the quotes.

.. Note::

   If you run into build issues while experimenting with build options, be sure
   to remove the ``_skbuild`` directory. If you had Arbor installed already,
   you may need to remove it first before you can (re)compile it with the flags you need.

The following flags can be used to configure the installation:

* ``ARB_WITH_MPI=<ON|OFF>``: Enable MPI support, requires MPI library. Default
  ``OFF``. If you intend to use ``mpi4py``, you need to install the package before
  building Arbor, as binding it requires access to its headers.
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

   There are more, advanced flags that can be set. We are using ``scikit-build``
   and ``CMake`` under the hood, so all flags and options valid in ``CMake`` can
   be used in this fashion.

   Allthough the
   `scikit-build documentation <https://scikit-build.readthedocs.io/en/latest/usage.html#environment-variable-configuration>`_
   mentions that you can also pass the build options with ``--install-option=""``,
   this will cause ``pip`` to build all dependencies, including all build-dependencies,
   instead of downloading them from PyPI.
   ``CMAKE_ARGS=""`` saves you the build time, and also downloading and setting up the dependencies they in turn require to be present.
   Setting ``CMAKE_ARGS=""`` is in addition compatible with build front-ends like `build <https://pypa-build.readthedocs.io>`_.

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

    CMAKE_ARGS="-DARB_WITH_MPI=ON" pip3 install ./arbor

**Compile with** :ref:`vectorization <install-vectorize>` on a system with a SkyLake
:ref:`architecture <install-architecture>`:

.. code-block:: bash

    CMAKE_ARGS="-DARB_VECTORIZE=ON -DARB_ARCH=skylake" pip3 install ./arbor
    
**Enable NVIDIA GPUs (compiled with nvcc)**. This requires the :ref:`CUDA toolkit <install-gpu>`:

.. code-block:: bash

    CMAKE_ARGS="-DARB_GPU=cuda" pip3 install ./arbor

**Enable NVIDIA GPUs (compiled with clang)**. This also requires the :ref:`CUDA toolkit <install-gpu>`:

.. code-block:: bash

    CMAKE_ARGS="-DARB_GPU=cuda-clang" pip3 install ./arbor

**Enable AMD GPUs (compiled with hipcc)**. This requires setting the ``CC`` and ``CXX``
:ref:`environment variables <install-gpu>`:

.. code-block:: bash

    CC=clang CXX=hipcc CMAKE_ARGS="-DARB_GPU=hip" pip3 install ./arbor

Note on performance
-------------------

The Python interface can incur significant memory and runtime overheads relative to C++
during the *model building* phase, however simulation performance is the same
for both interfaces.
