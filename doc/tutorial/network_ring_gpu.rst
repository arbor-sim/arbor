.. _tutorialgpu:

GPU and profiling
=================

In this example, the ring network created in an :ref:`earlier tutorial <tutorialnetworkring>` will be used to run the model with a GPU. In addition, it is shown how to profile the performance difference. Only the differences with that tutorial will be described.

.. Note::

   **Concepts covered in this example:**

   1. Building a :py:class:`arbor.context` that'll use a GPU.
      This requires that you have built Arbor with GPU support enabled.
   2. Build a :class:`arbor.domain_decomposition` and provide a :class:`arbor.partition_hint`.
   3. Profile an Arbor simulation using :class:`arbor.meter_manager`.

The hardware context
********************

An :ref:`execution context <modelcontext>` describes the hardware resources on which the simulation will run.
It contains the thread pool used to parallelise work on the local CPU, and optionally describes GPU resources
and the MPI communicator for distributed simulations. In some other examples, the :class:`arbor.single_cell_model`
object created the execution context :class:`arbor.context` behind the scenes. The details of the execution
context can be customized by the user. We may specify the number of threads in the thread pool; determine the
id of the GPU to be used; or create our own MPI communicator.

Step **(11)** creates a hardware context where we set the :py:attr:`~arbor.proc_allocation.gpu_id`. This requires 
that you have built Arbor manually, with GPU support (See :ref:`here <in_python_adv>` how to do that). On a regular 
consumer device with a single GPU, the index you should pass is ``0``. Change the value to run the example with and 
without GPU.

.. literalinclude:: ../../python/example/network_ring_gpu.py
   :language: python
   :lines: 123-127

Profiling
*********

Arbor comes with a :class:`arbor.meter_manager` to help you profile your simulations. In this case, you can run the 
example with ``gpu_id=None`` and ``gpu_id=0`` and observe the difference with the :class:`~arbor.meter_manager`.

Step **(12)** sets up the meter manager and starts it using the (only) context. This way, only Arbor related execution is measured, not Python code.

Step **(13)** instantiates the recipe and sets the first checkpoint on the meter manager. We now have the time it took to construct the recipe.

.. literalinclude:: ../../python/example/network_ring_gpu.py
   :language: python
   :lines: 129-136

The domain decomposition
************************

The domain decomposition describes the distribution of the cells over the available computational resources.
The :class:`arbor.single_cell_model` also handled that without our knowledge in the previous examples.
Now, we have to define it ourselves.

The :class:`arbor.domain_decomposition` class can be manually created by the user, by deciding which cells
go on which ranks. Or we can use a load balancer that can partition the cells across ranks according to
some rules. Arbor provides :class:`arbor.partition_load_balance`, which, using the recipe and execution
context, creates the :class:`arbor.domain_decomposition` object for us.

A way to customize :class:`arbor.partition_load_balance` is by providing a :class:`arbor.partition_hint`. They let
you configure how cells are distributed over the resources in the :class:`~arbor.context`, but without requiring you
to know the precise configuration of a :class:`~arbor.context` up front. Whether you run your simulation on your 
laptop CPU, desktop GPU, CPU cluster of GPU cluster, using :class:`partition hints<arbor.partition_hint>` you can
just say: use GPUs, if available. You only have to change the :class:`~arbor.context` to actually define which 
hardware Arbor will execute on.

Step **(14)** creates a :class:`arbor.partition_hint`, and tells it to put 1000 cells in a groups allocated to GPUs, 
and to prefer the utilisation of the GPU if present. In fact, the default distribution strategy of 
:class:`arbor.partition_load_balance` already spreads out cells as evenly as possible over CPUs, and groups
(up to 1000) on GPUs, so strictly speaking it was not necessary to give that part of the hint.
Lastly, a dictionary is created with which hints are assigned to a particular :class:`arbor.cell_kind`.
Different kinds may favor different execution, hence the option.
In this simulation, there are only :class:`arbor.cell_kind.cable`, so we assign the hint to that kind.

Step **(15)** creates a :class:`arbor.partition_load_balance` with the recipe, context and hints created above.
Another checkpoint will help us understand how long creating the load balancer took.

.. literalinclude:: ../../python/example/network_ring_gpu.py
   :language: python
   :lines: 138-148


The simulation
**************

Step **(16)** creates a :class:`arbor.simulation`, sets the spike recorders to record, creates a :term:`handle`
to their eventual results and makes another checkpoint.

.. literalinclude:: ../../python/example/network_ring_gpu.py
   :language: python
   :lines: 150-154

The execution
*************

Step **(17)** runs the simulation. Since we have more cells this time, which are connected in series,
it will take some time for the action potential to propagate. In the :ref:`ring network <tutorialnetworkring>`
we could see it takes about 5 ms for the signal to propagate through one cell, so let's set the runtime to
``5*ncells``. Then, another checkpoint, so that we'll know how long the simulation took.

.. literalinclude:: ../../python/example/network_ring_gpu.py
   :language: python
   :lines: 156-159

The results
***********

The scientific results should be similar, other than number of cells, to those in :ref:`ring network <tutorialnetworkring>`,
so we'll not discuss them here. Let's turn our attention to the :class:`~arbor.meter_manager`.

.. literalinclude:: ../../python/example/network_ring_gpu.py
   :language: python
   :lines: 161-163

Step **(18)** shows how :class:`arbor.meter_report` can be used to read out the :class:`~arbor.meter_manager`.
It generates a table with the time between checkpoints. As an example, the following table is the result of a run
on a 2019 laptop CPU:

::

   ---- meters -------------------------------------------------------------------------------
   meter                         time(s)      memory(MB)
   -------------------------------------------------------------------------------------------
   recipe-create                   0.000           0.059
   load-balance                    0.000           0.007
   simulation-init                 0.012           0.662
   simulation-run                  0.037           0.319
   meter-total                     0.049           1.048

The full code
*************

You can find the full code of the example at ``python/examples/network_ring_gpu.py``.
