.. _getstarted_python:

Python
======

Arbor provides access to all of the C++ library's functionality in Python,
which is the only interface for many users.
The getting started guides will introduce Arbor via the Python interface.

To test that Arbor is available, try the following in a `Python 3 <python2_>`_ interpreter:

.. code-block:: python

    >>> import arbor
    >>> print(arbor.__config__)
    {'mpi': True, 'mpi4py': True, 'gpu': False, 'version': '0.2.3-dev'}
    >>> print(arbor.__version__)
    0.2.3-dev

The dictionary ``arbor.__config__`` contains information about the Arbor installation.
This can be used to check that Arbor supports features that you require to run your model,
or even to dynamically decide how to run a model.
Single cell models like do not require parallelism like
that provided by MPI or GPUs, so the ``'mpi'`` and ``'gpu'`` fields can be ``False``.

Performance
--------------

The Python interface can incur significant memory and runtime overheads relative to C++
during the *model building* phase, however simulation performance is the same
for both interfaces.

.. _python2:

Python 2
----------

Python 2 reached `end of life <https://pythonclock.org/>`_ in January 2020.
Arbor only officially supports Python 3.6 and later, and all examples in the
documentation are in Python 3. While it is possible to install and run Arbor
using Python 2.7 by setting the ``PYTHON_EXECUTABLE`` variable when
:ref:`configuring CMake <install-python>`, support is only provided for using
Arbor with Python 3.6 and later.

Installing
-------------

Before starting Arbor needs to be installed with the Python interface enabled.
The easiest way to get started with the Python interface is to install Arbor using
`pip <https://packaging.python.org/tutorials/installing-packages>`_.

Installing with pip requires two steps:
**(1)** Obtain Arbor source code from GitHub;
**(2)** then use pip to compile and install the Arbor package in one shot.

.. code-block:: bash

    git clone https://github.com/arbor-sim/arbor.git --recursive
    # use pip (recommended)
    pip3 install ./arbor
    # use setuptools and python directly
    python3 install ./arbor/setup.py

This will install Arbor as a site-wide package with only multi-threading enabled.

To enable more advanced forms of parallelism, the following flags can optionally
be passed to the pip install command:

* ``--mpi``: enable mpi support (requires MPI library).
* ``--gpu``: enable nvidia cuda support (requires cudaruntime and nvcc).
* ``--vec``: enable vectorization. This might require carefully choosing the ``--arch`` flag.
* ``--arch``: cpu micro-architecture to target. By default this is set to ``native``.

If calling ``setup.py`` the flags must to come after ``install`` on the command line,
and if being passed to pip they must be passed via ``--install-option``. The examples
below demonstrate this for both pip and ``setup.py``, with pip being our recommend method.

Vanilla install with no additional options/features enabled:

.. code-block:: bash

    pip3 install ./arbor
    python3 ./arbor/setup.py install

Enable MPI support. This might require loading an MPI module or setting the ``CC`` and ``CXX``
:ref:`environment variables <install-mpi>`.

.. code-block:: bash

    pip3 install --install-option='--mpi' ./arbor
    python3 ./arbor/setup.py install --mpi

Compile with :ref:`vectorization <install-vectorize>` on a system with SkyLake
:ref:`architecture <install-architecture>`:

.. code-block:: bash

    pip3 install --install-option='--vec' --install-option='--arch=skylake' ./arbor
    python3 ./arbor/setup.py install --vec --arch=skylake

Compile with support for NVIDIA GPUs. This requires that the :ref:`CUDA toolkit <install-gpu>`
is installed and the CUDA compiler nvcc is available:

.. code-block:: bash

    pip3 install --install-option='--gpu' ./arbor
    python3 ./arbor/setup.py install --gpu

.. Note::
    Installation takes a while because pip has to compile the Arbor C++ library and
    wrapper, which takes a few minutes. Pass the ``--verbose`` flag to pip
    to see the individual steps being preformed if concerned that progress
    is halting.

.. Note::
    Detailed instructions on how to install using CMake are in the
    :ref:`Python configuration <install-python>` section of the
    :ref:`installation guide <installarbor>`.
    CMake is recommended for developers, integration with package managers such as
    Spack and EasyBuild, and users who require fine grained control over compilation
    and installation.

.. Note::
    If there is an error installing with pip you want to report,
    run pip with the ``--verbose`` flag, and attach the output (along with
    the pip command itself) to a ticket on the
    `Arbor GitHub <https://github.com/arbor-sim/arbor/issues>`_.
    For example, ``pip3 install --install-option='--mpi' --verbose .``.

Dependencies
^^^^^^^^^^^^^

If a downstream dependency of Arbor that requires Arbor be built with
a specific feature enabled, use ``requirements.txt`` to
`define the constraints <https://pip.pypa.io/en/stable/reference/pip_install/#per-requirement-overrides>`_.
For example, a package that depends on `arbor` would version 0.3 or later
with MPI support would add the following to its requirements.

.. code-block:: python

    arbor >= 0.3 --install-option='--gpu' \
                 --install-option='--mpi'

