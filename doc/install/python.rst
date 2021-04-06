.. _in_python:

Python Installation
===================

Arbor's Python API will be the most convenient interface for most users.

.. note::
    Arbor requires Python version 3.6 and later. It is advised that you update `pip` as well.

Getting Arbor
-------------

Every point release of Arbor is pushed to the Python Package Index. The easiest way to get Arbor is with
`pip <https://packaging.python.org/tutorials/installing-packages>`_:

.. code-block:: bash

    pip3 install arbor

.. note::
    You will need to have some development packages installed in order to build Arbor this way.

    * Ubuntu/Debian: `sudo apt install git build-essential python3-dev python3-pip libxml2-dev`
    * Fedora/CentOS/Red Hat: `sudo yum install git @development-tools python3-devel python3-pip libxml2-devel`
    * macOS: get `brew` `here <https://brew.sh>`_ and run `brew install cmake clang python3 libxml2`
    * Windows: the simplest way is to use `WSL <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ and then follow the instructions for Ubuntu.

If you wish to get the latest Arbor straight from
the master branch in our git repository, you can run:

.. code-block:: bash

    pip3 install git+https://github.com/arbor-sim/arbor.git

To test that Arbor is available, try the following in a Python interpreter
to see information about the version and enabled features:

.. code-block:: python

    >>> import arbor
    >>> print(arbor.__version__)
    >>> print(arbor.__config__)

You are now ready to use Arbor! You can continue reading these documentation pages, have a look at the
:ref:`Python API reference<pyoverview>` , or visit the :ref:`Quick Start page<tutorialsinglecell>`.

.. Note::
    To get help in case of problems installing with pip, run pip with the ``--verbose`` flag, and attach the output
    (along with the pip command itself) to a ticket on `Arbor's issues page <https://github.com/arbor-sim/arbor/issues>`_.

Customising Arbor
^^^^^^^^^^^^^^^^^

If you want to work on Arbor's code, you can get a copy of our repo and point `pip` at the local directory:

.. code-block:: bash

    # get your copy of the Arbor source
    git clone https://github.com/arbor-sim/arbor.git --recursive
    # make your changes and then instruct pip to build and install the local source
    pip3 install ./arbor/

Every time you make changes to the code, you'll have to repeat the second step.

Advanced options
^^^^^^^^^^^^^^^^

By default Arbor is installed with multi-threading enabled.
To enable more advanced forms of parallelism, the following optional flags can
be used to configure the installation:

* ``--mpi``: Enable MPI support (requires MPI library).
* ``--gpu``: Enable GPU support for NVIDIA GPUs with nvcc using ``cuda``, or with clang using ``cuda-clang`` (both require cudaruntime).
  Enable GPU support for AMD GPUs with hipcc using ``hip``. By default set to ``none``, which disables gpu support.
* ``--vec``: Enable vectorization. This might require choosing an appropriate architecture using ``--arch``. Note that on x86-64 platforms compilation will fail if you enable vectorization, but the CPU or ``--arch`` does not support any form of AVX.
* ``--arch``: CPU micro-architecture to target. By default this is set to ``native``.

If calling ``setup.py`` the flags must come after ``install`` on the command line,
and if being passed to pip they must be passed via ``--install-option``. The examples
below demonstrate this for both pip and ``setup.py``.

**Vanilla install** with no additional features enabled:

.. code-block:: bash

    pip3 install arbor
    python3 ./arbor/setup.py install

**With MPI support**. This might require loading an MPI module or setting the ``CC`` and ``CXX``
:ref:`environment variables <install-mpi>`:

.. code-block:: bash

    pip3 install --install-option='--mpi' ./arbor
    python3 ./arbor/setup.py install --mpi

**Compile with** :ref:`vectorization <install-vectorize>` on a system with a SkyLake
:ref:`architecture <install-architecture>`:

.. code-block:: bash

    pip3 install --install-option='--vec' --install-option='--arch=skylake' arbor
    python3 ./arbor/setup.py install --vec --arch=skylake

**Enable NVIDIA GPUs (compiled with nvcc)**. This requires the :ref:`CUDA toolkit <install-gpu>`:

.. code-block:: bash

    pip3 install --install-option='--gpu=cuda' ./arbor
    python3 ./arbor/setup.py install  --gpu=cuda

**Enable NVIDIA GPUs (compiled with clang)**. This also requires the :ref:`CUDA toolkit <install-gpu>`:

.. code-block:: bash

    pip3 install --install-option='--gpu=cuda-clang' ./arbor
    python3 ./arbor/setup.py install --gpu=cuda-clang

**Enable AMD GPUs (compiled with hipcc)**. This requires setting the ``CC`` and ``CXX``
:ref:`environment variables <install-gpu>`

.. code-block:: bash

    pip3 install --install-option='--gpu=hip' ./arbor
    python3 ./arbor/setup.py install --gpu=hip

.. Note::
    Setuptools compiles the Arbor C++ library and
    wrapper, which can take a few minutes. Pass the ``--verbose`` flag to pip
    to see the individual steps being performed if you are concerned that progress
    is halting.

.. Note::
    Detailed instructions on how to install using CMake are in the
    :ref:`Python configuration <install-python>` section of the
    :ref:`installation guide <in_build_install>`.
    CMake is recommended for developers, integration with package managers such as
    Spack and EasyBuild, and fine-grained control over compilation and installation.

Dependencies
^^^^^^^^^^^^

If a downstream dependency requires Arbor be built with
a specific feature enabled, use ``requirements.txt`` to
`define the constraints <https://pip.pypa.io/en/stable/reference/pip_install/#per-requirement-overrides>`_.
For example, a package that depends on `arbor` version 0.3 or later
with MPI support would add the following to its requirements:

.. code-block:: python

    arbor >= 0.3 --install-option='--gpu=cuda' \
                 --install-option='--mpi'

Note on performance
-------------------

The Python interface can incur significant memory and runtime overheads relative to C++
during the *model building* phase, however simulation performance is the same
for both interfaces.
