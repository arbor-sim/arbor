.. _pyoverview:

Python
=========

The Python frontend for Arbor is an interface that the vast majority of users will use to interact with Arbor.
This section covers how to use the frontend with examples and detailed descriptions of features.

.. Note::
    If you haven't set up Arbor yet, see the :ref:`Python installation guide <in_python>`.

.. _simsteps:

.. rubric:: Simulation steps

The workflow for defining and running a model defined in :ref:`modelsimulation` can be performed
in Python as follows:

1. Describe the neuron model by defining an :class:`arbor.recipe`;
2. Describe the computational resources to use for simulation using :class:`arbor.proc_allocation` and :class:`arbor.context`;
3. Partition the model over the hardware resources using :class:`arbor.partition_load_balance`;
4. Run the model by initiating then running the :class:`arbor.simulation`.

These details are described and examples are given in the next sections :ref:`pycell`, :ref:`pyrecipe`, :ref:`pydomdec`, :ref:`pysimulation`, and :ref:`pyprofiler`.

.. note::

    Detailed information on Arbor's Python features can also be obtained with Python's ``help`` function, e.g.

    .. code-block:: python3

        >>> help(arbor.proc_allocation)
        Help on class proc_allocation in module arbor:

        class proc_allocation(pybind11_builtins.pybind11_object)
        |  Enumerates the computational resources on a node to be used for simulation.
        |...

.. toctree::
   :caption: Python API:
   :maxdepth: 2

   recipe
   cell
   interconnectivity
   hardware
   domdec
   simulation
   profiler
   cable_cell
   lif_cell
   spike_source_cell
   benchmark_cell
   single_cell_model
