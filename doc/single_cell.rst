.. _single_cell:

Single Cell Models
==================

This page is a practical guide for beginners on how to make and run single cell Arbor model in Python.

TODO: this should be designed to help people who are familiar with NEURON.
    - We could even provide a custom *Note* style box called something like *Porting* or *NEURON* to highlight differences or
      subtleties relative to NEURON as they arise.

Python
------

Arbor is a C++ library, designed for integration into simulation packages and high performance computing.
Arbor's Python interface gives access to all of the C++ library's functionality from Python.

The Python interface can incur significant performance overheads relative to C++ during the *model building* phase,
however simulation performance will be the same for both interfaces.

.. Note::
    To try this tutorial Arbor needs to be installed with the Python interface enabled.
    See the `installation guide <_installarbor>`_ if you have not installed Arbor yet.

.. Warning::
    Python 2.7 will reach `end of life <https://pythonclock.org/>`_ in 2020.
    Arbor should be installed using Python 3, and all examples in the documentation are in
    Python 3. It might be possible to install and run Arbor using Python 2.7, however Arbor support
    will only be provided for Python 3.

To test that Arbor is available, open Python and try the following:

.. container:: example-code

    .. code-block:: python

        import arbor
        arbor.config()

        {'mpi': True, 'mpi4py': True, 'gpu': False, 'version': '0.3'}

Calling ``arbor.config()`` returns a dictionary with information about the Arbor installation.
This can be used to check that Arbor supports features that you require to run your model,
or even to dynamically decide how to run a model.
To run single cell models like the one introduced here we don't need parallelism like
that provided by MPI or GPUs, so the ``'mpi'`` and ``'gpu'`` fields can be ``False``.

Morphology
----------

building the morphology there are two approaches: construct it manually using
``sample_tree`` or ``flat_cell_builder``, or load from swc file.

Question: cover all of these here?
    - we could just ``flat_cell_builder`` because it is most comfortable for
      users coming over from NEURON.


