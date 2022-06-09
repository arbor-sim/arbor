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
   single_cell_recipe
   single_cell_detailed
   single_cell_detailed_recipe
   single_cell_cable

Networks
--------

.. toctree::
   :maxdepth: 1

   network_ring
   network_ring_mpi
   single_cell_allen
   two_cells_gap_junctions

Probes
------

.. toctree::
   :maxdepth: 1

   tutorial_lfpykit
