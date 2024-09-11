.. _tutorialsinglecellswcrecipe:

A detailed single-cell recipe
=============================

This example builds the same single-cell model as :ref:`tutorialsinglecellswc`,
except using a :class:`arbor.recipe` and :class:`arbor.simulation` instead of a :class:`arbor.single_cell_model`.

This time, we'll learn a bit more about setting up advanced features using a :class:`arbor.recipe`.

.. Note::

   **Concepts covered in this example:**

   1. Building a :class:`arbor.recipe`.
   2. Building an :class:`arbor.context`.
   3. Creating a :class:`arbor.simulation`.
   4. Running the simulation and visualising the results.

The cell
********

We reuse the cell construction code from the :ref:`original example
<tutorialsinglecellswc-cell>` where it is explained in detail.
Constructing cells outside the recipe is not required, but may be
convenient and potentially faster if many copies of the same cell
are required.

The recipe
**********

The :class:`arbor.single_cell_model` used in the original example created an
:class:`arbor.recipe` under the hood, and abstracted away the details so we were
unaware of its existence. In this example, we will examine the recipe in detail:
how to create one and why it is needed.

.. literalinclude:: ../../python/example/single_cell_detailed_recipe.py
   :language: python
   :lines: 78-128

Let's go through the recipe point by point.

Step **(5.1)** defines the class constructor. As per :class:`arbor.recipe`
instructions, we call ``arbor.recipe.__init__(self)`` as the very first thing.
This is to ensure correct initialization.

We then create the ``self.the_props`` variable. This will hold the global
properties of the model, which apply to all the cells in the network. We
initialize it with :class:`arbor.cable_global_properties`, which comes with the
``default`` mechanism catalogue built-in. We set all the properties of the
system similar to what we did in the :ref:`original example
<tutorialsinglecellswc-gprop>`. One last important step is to extend
``self.the_props`` to include the Allen catalogue, because it holds the *Ih*
mechanism. The *hh* and *pas* mechanisms came with the default catalogue.

Step **(5.2)** overrides the :meth:`~arbor.recipe.num_cells` method. It takes no
arguments and returns the count of cells in the simulation. The global id (gid)
of cells runs between 0 and the value returned from here and is used to query
recipes for cell descriptions. Technically, this method doesn't need overriding,
but the default is zero, resulting in an empty simulation.

Step **(5.3)** overrides the :meth:`~arbor.recipe.cell_kind` method. It takes
one argument: ``gid``. Given the gid, this method returns the kind of the cell.
Our defined cell is a :class:`arbor.cell_kind.cable`, so we simply return that.
Arbor uses the kind to determine what description is expected from ``cell_description``;
if the two do not match, an error will occur.

Step **(5.4)** overrides the :meth:`~arbor.recipe.cell_description` method. It
takes one argument: ``gid``. Given the gid, this method returns the cell
description which is the cell object passed to the constructor of the recipe. We
return ``cell``, the cell created just above.

.. note::

   While splitting the kind and description into two methods may seem redundant,
   it allows Arbor to optimize the simulation layout before constructing any
   cells.

Step **(5.5)** overrides the :meth:`~arbor.recipe.probes` method. It takes one
argument: ``gid``. Given the gid, this method returns all the probes on the
cell. The probes can be of many different kinds, measuring different quantities
at different locations of the cell. Like in the original example, we will create
the voltage probe at the ``"custom_terminal"`` locset. This probe was registered
directly using the :class:`arbor.single_cell_model` object. Now, it has to be
explicitly created and registered in the recipe.

Step **(5.6)** overrides the :meth:`~arbor.recipe.global_properties` method. It
takes one argument: ``kind``. This method returns the default global properties
of the model which apply to all cells in the network of that kind. We only use
``cable`` cells in this example (but there are more) and thus always return a
``cable_cell_properties`` object. We return ``self.the_props`` which we defined
in step **(1)**.

.. literalinclude:: ../../python/example/single_cell_detailed_recipe.py
   :language: python
   :lines: 129-131

Now we can instantiate a ``single_recipe`` object.

The simulation
**************

We have all we need to create an :class:`arbor.simulation` object.

Before we run the simulation, however, we need to register what results we expect once execution is over.
This was handled by the :class:`arbor.single_cell_model` object in the original example.

.. literalinclude:: ../../python/example/single_cell_detailed_recipe.py
   :language: python
   :lines: 133-140

We would like to get a list of the spikes on the cell during the runtime of the
simulation, and we would like to plot the voltage registered by the probe on the
"custom_terminal" locset. Without the call to ``sample``, the probe will be
present, but no data will recorded. This can help to toggle different probes and
possibly save some memory on the unsampled ones. Sampling requires a unique
identifier for the probe we want to attach to, which consists of the gid of the
cell and the label we gave to the ``place`` method when setting the probe. Each
sampler is identified by an opaque handle that can be used to retrieve the
recorded data.

We can now run the simulation we just instantiated for a duration of 100 ms with
a time step of 0.025 ms.

.. literalinclude:: ../../python/example/single_cell_detailed_recipe.py
   :language: python
   :lines: 141-142

The results
***********

The last step is result collection. We instructed the simulation to record the
spikes on the cell, and to sample the probe.

We can print the times of the spikes:

.. literalinclude:: ../../python/example/single_cell_detailed_recipe.py
   :language: python
   :lines: 144-148

The probe results, again, warrant some more explanation:

.. literalinclude:: ../../python/example/single_cell_detailed_recipe.py
   :language: python
   :lines: 139

``sim.samples`` takes a ``handle`` that is associated with the probe we wish to
examine. These opaque objects are returned from the calls to ``sim.sample``.
Each call returns a list of ``(data, meta)`` objects. Here, ``meta`` describes
the location set of the probe, which can be a single place or a list of places.
The other item -- ``data`` -- is a Numpy array comprising one column for the
time and one for each entry in the location list. The size of the returned list
depends on the number of discrete locations pointed to by the handle. We placed
the probe on the "custom_terminal" locset which is represented by 2 locations on
the morphology. We therefore expect the length of ``sim.samples(handle)`` to
be 2.

We plot the results using pandas and seaborn as we did in the original example,
and expect the same results:

.. literalinclude:: ../../python/example/single_cell_detailed_recipe.py
   :language: python
   :lines: 150-

The following plot is generated. Identical to the plot of the original example.

.. figure:: single_cell_detailed_result.svg
    :width: 400
    :align: center


The full code
*************

You can find the full code of the example at
``python/examples/single_cell_detailed_recipe.py``.
