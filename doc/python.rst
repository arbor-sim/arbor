.. _getstarted_python:

Python
======

Arbor's Python wrapper will be the most convenient interface for most users.

Getting Arbor
-------------

The easiest way to get Arbor is with
`pip <https://packaging.python.org/tutorials/installing-packages>`_:

.. code-block:: bash

    pip3 install arbor

It is also possible to use Setuptools directly on a local copy of the source code:

.. code-block:: bash

    # use setuptools and python directly
    git clone https://github.com/arbor-sim/arbor.git --recursive
    python3 install ./arbor/setup.py

.. note::
    Arbor's Setuptools process simplifies installation for common configurations
    on laptops and workstations by calling CMake under the hood.

    To install Arbor on a HPC cluster, or to configure Arbor with system-specific
    options, we recommend using the :ref:`CMake build process <installarbor>`.

To test that Arbor is available in Python, try the following in a `Python 3 <python2_>`_ interpreter
to see information about the version and enabled features:

.. code-block:: python

    >>> import arbor
    >>> print(arbor.__version__)
    >>> print(arbor.__config__)

Advanced Options
^^^^^^^^^^^^^^^^^^

By default Arbor is installed with multi-threading enabled.
To enable more advanced forms of parallelism, the following optional flags can
be used to configure the installation:

* ``--mpi``: Enable MPI support (requires MPI library).
* ``--gpu``: Enable NVIDIA CUDA support (requires cudaruntime and nvcc).
* ``--vec``: Enable vectorization. This might require choosing an appropriate architecture using ``--arch``.
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

Compile with :ref:`vectorization <install-vectorize>` on a system with SkyLake
:ref:`architecture <install-architecture>`:

.. code-block:: bash

    pip3 install --install-option='--vec' --install-option='--arch=skylake' arbor
    python3 ./arbor/setup.py install --vec --arch=skylake

**Enable NVIDIA GPUs**. This requires the :ref:`CUDA toolkit <install-gpu>`:

.. code-block:: bash

    pip3 install --install-option='--gpu' ./arbor
    python3 ./arbor/setup.py install --gpu

.. Note::
    Setuptools compiles the Arbor C++ library and
    wrapper, which can take a few minutes. Pass the ``--verbose`` flag to pip
    to see the individual steps being performed if you are concerned that progress
    is halting.

.. Note::
    Detailed instructions on how to install using CMake are in the
    :ref:`Python configuration <install-python>` section of the
    :ref:`installation guide <installarbor>`.
    CMake is recommended for developers, integration with package managers such as
    Spack and EasyBuild, and users who require fine grained control over compilation
    and installation.

.. Note::
    To report problems installing with pip,
    run pip with the ``--verbose`` flag, and attach the output (along with
    the pip command itself) to a ticket on
    `Arbor's issues page <https://github.com/arbor-sim/arbor/issues>`_.

Dependencies
^^^^^^^^^^^^^

If a downstream dependency requires Arbor be built with
a specific feature enabled, use ``requirements.txt`` to
`define the constraints <https://pip.pypa.io/en/stable/reference/pip_install/#per-requirement-overrides>`_.
For example, a package that depends on `arbor` version 0.3 or later
with MPI support would add the following to its requirements:

.. code-block:: python

    arbor >= 0.3 --install-option='--gpu' \
                 --install-option='--mpi'

Performance
--------------

The Python interface can incur significant memory and runtime overheads relative to C++
during the *model building* phase, however simulation performance is the same
for both interfaces.

.. _python2:

Python 2
----------

Python 2 reached `end of life <https://pythonclock.org/>`_ in January 2020.
Arbor only provides support for Python 3.6 and later.

.. note::
    It might be possible to install and run Arbor
    using Python 2.7 by setting the ``PYTHON_EXECUTABLE`` variable when
    :ref:`configuring CMake <install-python>`.
    However, Arbor is not tested against Python 2.7, and we won't be able
    to provide support.


