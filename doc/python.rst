.. _getstarted_python:

Python
======

Arbor provides access to all of the C++ library's functionality in Python,
which is the only interface for many users.
The getting started guides will introduce Arbor via the Python interface.

Before starting Arbor needs to be installed with the Python interface enabled,
as per the `installation guide <_installarbor>`_.
To test that Arbor is available, open `Python 3 <python2_>`_, and try the following:

.. container:: example-code

    .. code-block:: python

        import arbor
        arbor.config()

        {'mpi': True, 'mpi4py': True, 'gpu': False, 'version': '0.3'}

Calling ``arbor.config()`` returns a dictionary with information about the Arbor installation.
This can be used to check that Arbor supports features that you require to run your model,
or even to dynamically decide how to run a model.
Single cell models like the one introduced here we do not require parallelism like
that provided by MPI or GPUs, so the ``'mpi'`` and ``'gpu'`` fields can be ``False``.

Performance
--------------

The Python interface can incur significant performance overheads relative to C++
during the *model building* phase, however simulation performance will be the same
for both interfaces.

.. _python2:

Python 2
----------

Python 2.7 will reach `end of life <https://pythonclock.org/>`_ in January 2020.
Arbor should be installed using Python 3, and all examples in the documentation are in
Python 3. It might be possible to install and run Arbor using Python 2.7, however Arbor support
will only be provided for Python 3.

NEURON users that use Python 2 can take the opportunity of trying Arbor to make
the move to Python 3.

