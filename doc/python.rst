.. _getstarted_python:

Python
======

Arbor provides access to all of the C++ library's functionality in Python,
which is the only interface for many users.
The getting started guides will introduce Arbor via the Python interface.

To test that Arbor is available, try the following in a `Python 3 <python2_>`_ interpreter:

.. container:: example-code

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

Installing
-------------

Before starting Arbor needs to be installed with the Python interface enabled,
following the :ref:`Python configuration <pythonfrontend>` in
the
:ref:`installation guide <installarbor>`.


Performance
--------------

The Python interface can incur significant performance overheads relative to C++
during the *model building* phase, however simulation performance is be the same
for both interfaces.

.. _python2:

Python 2
----------

Python 2 reached `end of life <https://pythonclock.org/>`_ in January 2020.
Arbor only officially supports Python 3.6 and later, and all examples in the
documentation are in Python 3. While it is possible to install and run Arbor
using Python 2.7 by setting the ``PYTHON_EXECUTABLE`` variable when
configuring CMake, support is only provided for using Arbor with Python 3.6
and later.

