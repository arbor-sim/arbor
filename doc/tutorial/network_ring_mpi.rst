.. _tutorialmpi:

Distributed ring network (MPI)
==============================

In this example, the ring network created in an :ref:`earlier tutorial <tutorialnetworkring>` will be used to run the model in
a distributed context using MPI. Only the differences with that tutorial will be described.

.. Note::

   **Concepts covered in this example:**

   1. Building a basic MPI aware :py:class:`arbor.context` to run a network.
      This requires that you have built Arbor with MPI support enabled.
   2. Running the simulation and extracting the results.

The recipe
**********

Step **(11)** is changed to generate a network with five hundred cells.

.. literalinclude:: ../../python/example/network_ring_mpi.py
   :language: python
   :lines: 124-126

The hardware context
********************

An :ref:`execution context <modelcontext>` describes the hardware resources on which the simulation will run.
It contains the thread pool used to parallelise work on the local CPU, and optionally describes GPU resources
and the MPI communicator for distributed simulations. In some other examples, the :class:`arbor.single_cell_model`
object created the execution context :class:`arbor.context` behind the scenes. The details of the execution
context can be customized by the user. We may specify the number of threads in the thread pool; determine the
id of the GPU to be used; or create our own MPI communicator.

The configuration of the context will need to be changed to reflect the change in hardware.
First of all, we scrap setting `threads="avail_threads"` and instead use 
`MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface#Overview>`_ to distribute the work over nodes, cores and threads.

Step **(12)** uses the Arbor-built-in :py:class:`MPI communicator <arbor.mpi_comm>`, which is identical to the
``MPI_COMM_WORLD`` communicator you'll know if you are familiar with MPI. The :py:class:`arbor.context` takes a
communicator for its ``mpi`` parameter. Note that you can also pass in communicators created with ``mpi4py``.
We print both the communicator and context to observe how Arbor configures their defaults.

Step **(13)** creates the simulation using the recipe and the context created in the previous step.

.. literalinclude:: ../../python/example/network_ring_mpi.py
   :language: python
   :lines: 128-136

The execution
*************

Step **(16)** runs the simulation. Since we have more cells this time, which are connected in series, it will take some time for the action potential to propagate. In the :ref:`ring network <tutorialnetworkring>` we could see it takes about 5 ms for the signal to propagate through one cell, so let's set the runtime to ``5*ncells``.

.. literalinclude:: ../../python/example/network_ring_mpi.py
   :language: python
   :lines: 145-147

An important change in the execution is how the script is run. Whereas normally you run the Python script by passing
it as an argument to the ``python`` command, you need to use ``srun`` or ``mpirun`` (depending on your MPI
distribution) to execute a number of jobs in parallel. You can still execute the script using ``python``, but then
MPI will not execute on more than one node.

From the commandline, we can run the script using ``mpirun`` (``srun`` on clusters operated with SLURM) and specify the number of ranks (``NRANKS``)
or nodes. Arbor will spread the cells evenly over the ranks, so with ``NRANKS`` set to 5, we'd be spreading the 500
cells over 5 nodes, simulating 100 cells each.

.. code-block::

   mpirun -n NRANKS python mpi.py

The results
***********

Before we execute the simulation, we have to understand how Arbor distributes the computational load over the ranks.
After executing ``mpirun``, all nodes will run the same script. In the domain decomposition step, the nodes will use
the provided MPI communicator to divide the work. Once :py:func:`arbor.simulation.run` starts, each node will work on
their allocated cell ``gid`` s.

This is relevant for the collection of results: these are not gathered for you. Remember that in step **(15)** we
store the handles to the probes; these referred to particular ``gid`` s. The ``gid`` s are now distributed, so on one
node, the script will not find the cell referred to by the handle and therefore return an empty list (no results were found).

In step **(17)** we check, for each ``gid``, if the list returned by :py:func:`arbor.simulation.samples` has a nonzero
length. The effect is that we collect the results generated on this particular node. Since we now have ``NRANKS``
instances of our script, and we can't access the results between nodes, we have to write the results to disk and
analyse them later. We query :py:attr:`arbor.context.rank` to generate a unique filename for the result.

.. literalinclude:: ../../python/example/network_ring_mpi.py
   :language: python
   :lines: 149-

In a second script, ``network_ring_mpi_plot.py``, we load the results stored to disk into a pandas table, and plot the concatenated table as before:

.. literalinclude:: ../../python/example/network_ring_mpi_plot.py
   :language: python

To avoid an overcrowded plot, this plot was generated with just 50 cells:

.. figure:: network_ring_mpi_result.svg
    :width: 400
    :align: center

The full code
*************

You can find the full code of the example at ``python/examples/network_ring_mpi.py`` and ``python/examples/network_ring_mpi_plot.py``.
