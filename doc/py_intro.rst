.. _pyoverview:

Overview
=========
This section gives insights to the usage of Arbor's python front end :py:mod:`arbor` with examples and detailed descriptions of features.
The python front end is the main interface through which Arbor is used.

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

* first, **describe** the neuron model by defining a recipe;
* then, get the local **resources**, the **execution context**, and partition the **load balance**;
* finally, **execute** the model by initiating and running the simulation.

.. In order to visualise the result a **spike recorder** can be used and to analyse Arbor's performance a **meter manager** is available.

These steps are described and examples are given in the next subsections :ref:`pyCommon`, :ref:`pyRecipe`, :ref:`pydomdec` and :ref:`pysimulation`.

.. note::

    Detailed information on Arbor's python features can be obtained with the ``help`` function, e.g.

    >>> help(arbor)
