.. _pyoverview:

Overview
=========
This section gives insights to the usage of Arbor's python front end ``arbor`` with examples and detailed descriptions of features.
The python front end is the main interface through which Arbor is used.

.. _prerequisites:

Prerequisites
~~~~~~~~~~~~~

Once Arbor is built in the folder ``path/to/arbor/build`` (and/ or installed to ``path/to/arbor/install``, see the :ref:`installarbor` documentation) python needs to be set up by setting

.. code-block:: bash

    export PYTHONPATH="path/to/arbor/build/lib:$PYTHONPATH"

or, in case of installation

.. code-block:: bash

    export PYTHONPATH="path/to/arbor/install/lib/python3.6/site-packages:$PYTHONPATH"

With this setup, the Arbor python module ``arbor`` can be imported with python3 via

    >>> import arbor

.. _simsteps:

Simulation steps
~~~~~~~~~~~~~~~~

Then, according to the :ref:`modelsimulation` description the Arbor python module can be utilized to

* first, **describe** the neuron model by defining a recipe;
* then, get the local **resources**, the **execution context**, and partition the **load balance**;
* finally, **execute** the model by initiating and running the simulation.

In order to visualise the result a **spike recorder** can be used and to analyse Arbor's performance a **meter manager** is available.

These steps are described and examples are given in the next subsections :ref:`pycommon`, :ref:`pyrecipe`, :ref:`pydomdec` and :ref:`pysimulation`.

.. note::

    Detailed information on Arbor's python features can be obtained with the ``help`` function, e.g.

    >>> help(arbor.recipe)
