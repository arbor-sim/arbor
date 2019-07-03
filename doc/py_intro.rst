.. _pyoverview:

Overview
=========
The Python frontend for Arbor is an interface that the vast majority of users will use to interact with Arbor.
This section covers how to use the frontend with examples and detailed descriptions of features.

.. _prerequisites:

Prerequisites
~~~~~~~~~~~~~

Once Arbor has been built and installed (see the :ref:`installation guide <installarbor>`),
the location of the installed module needs to be set in the ``PYTHONPATH`` environment variable.
For example:

.. code-block:: bash

    export PYTHONPATH="/usr/lib/python3.7/site-packages/:$PYTHONPATH"

With this setup, Arbor's python module :py:mod:`arbor` can be imported with python3 via

    >>> import arbor

.. _simsteps:

Simulation steps
~~~~~~~~~~~~~~~~

The workflow for defining and running a model defined in :ref:`modelsimulation` can be performed
in Python as follows:

1. Describe the neuron model by defining an :class:`arbor.recipe`;
2. Describe the computational resources to use for simulation using :class:`arbor.proc_allocation` and :class:`arbor.context`;
3. Partition the model over the hardware resources using :class:`arbor.partition_load_balance`;
4. Run the model by initiating then running the :class:`arbor.simulation`.

These details are described and examples are given in the next sections :ref:`pycommon`, :ref:`pyrecipe`, :ref:`pydomdec`, :ref:`pysimulation`, and :ref:`pyprofiler`.

.. note::

    Detailed information on Arbor's python features can also be obtained with Python's ``help`` function, e.g.

    .. code-block:: python3

        >>> help(arbor.proc_allocation)
        Help on class proc_allocation in module arbor:

        class proc_allocation(pybind11_builtins.pybind11_object)
        |  Enumerates the computational resources on a node to be used for simulation.
        |...
