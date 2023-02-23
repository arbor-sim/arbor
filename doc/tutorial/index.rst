.. _tutorial:

Tutorials
=========

Grouped loosely by primary (but not exclusive!) focus, we have a set of tutorials to help you learn by doing. 

You can find some examples of full Arbor simulations in the ``python/examples`` directory of the
`Arbor repository <https://github.com/arbor-sim/arbor>`_.

The examples use ``pandas``, ``seaborn`` and ``LFPykit`` for analysis and plotting which are expected to be
installed independently from Arbor.

In an interactive Python interpreter, you can use ``help()`` on any class or function to get its
documentation. (Try, ``help(arbor.simulation)``, for example).

Cells
-----

.. toctree::
   :maxdepth: 1

   single_cell_model
   single_cell_detailed
   single_cell_cable
   single_cell_allen
   single_cell_bluepyopt

Recipes
-------

.. toctree::
   :maxdepth: 1
   
   single_cell_recipe
   single_cell_detailed_recipe

Networks
--------

.. toctree::
   :maxdepth: 1

   network_ring
   network_two_cells_gap_junctions

Probes
------

.. toctree::
   :maxdepth: 1

   probe_lfpykit

Stochastic Mechanisms
---------------------

.. toctree::
   :maxdepth: 1

   calcium_stdp_curve

Hardware
--------

.. toctree::
   :maxdepth: 1

   network_ring_mpi
   network_ring_gpu

Demonstrations
--------------

We try to collect models scientists have built in our `contributor space <https://github.com/arbor-contrib/>`_.
In addition to the tutorials, browsing these models should give you a good idea of what's possible with Arbor
and find get in contact with other Arbor users.
