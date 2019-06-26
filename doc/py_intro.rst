.. _pyoverview:

Overview
=========
This section gives insights to the usage of Arbor's python frontend :py:mod:`arbor` with examples and detailed descriptions of features.
The python frontend is the main interface through which Arbor is used.

.. _prerequisites:

Prerequisites
~~~~~~~~~~~~~

Once Arbor is built in the folder ``path/to/arbor/build`` (and/ or installed to ``path/to/arbor/install``, see the :ref:`installarbor` documentation) python needs to be set up by setting

.. code-block:: bash

    export PYTHONPATH="path/to/arbor/build/lib:$PYTHONPATH"

or, in case of installation

.. code-block:: bash

    export PYTHONPATH="path/to/arbor/install/lib/python3/site-packages:$PYTHONPATH"

With this setup, Arbor's python module :py:mod:`arbor` can be imported with python3 via

    >>> import arbor

.. _simsteps:

Simulation steps
~~~~~~~~~~~~~~~~

Then, according to the :ref:`modelsimulation` description Arbor's python module :py:mod:`arbor` can be utilized to

* first, describe the neuron model by defining an :class:`arbor.recipe`;
* then, get the local resources (:class:`arbor.proc_allocation`), the execution :class:`arbor.context`, and :class:`arbor.partition_load_balance`;
* finally, execute the model by initiating and running the :class:`arbor.simulation`.

In order to visualise the result a spike recorder can be attached using :func:`arbor.attach_spike_recorder`.
To analyse Arbor's performance an :class:`arbor.meter_manager` is available generating a measurement summary using :func:`arbor.make_meter_report`.

These details are described and examples are given in the next sections :ref:`pycommon`, :ref:`pyrecipe`, :ref:`pydomdec`, :ref:`pysimulation`, and :ref:`pyprofiler`.

.. note::

    Detailed information on Arbor's python features can also be obtained with Python's ``help`` function, e.g.

    >>> help(arbor)
