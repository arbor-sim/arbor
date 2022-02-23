.. _shared_state:

Shared State
============

The ``shared_state`` classes are collections to store backend-specific simulator
state and can be found in the ``backend/{multicore, gpu}`` directories.
Functionality manipulating such items gets delegated to the implementations of
the ``shared_state`` interface.

Ions
----

Ion state is stored as a series of arrays corresponding to NMODL variables.
These arrays have entries per CV and mechanisms need to go through the
``node_index_`` array to map the mechanism's internal CV index to the cell
group's CV index.

=======  ======= ===============================
Field    NMODL   Ion Property
=======  ======= ===============================
``iX_``  ``ica`` Current density
``eX_``  ``eca`` Reversal potential
``Xi_``  ``cai`` Internal concentration
``Xo_``  ``cao`` External concentration
=======  ======= ===============================

This table shows the mapping between NMODL -- for the ``ca`` ion species -- and
the ``ion_state`` members. The class is responsible for reseting currents and
concentrations.

Mechanisms
----------

All mechanisms' privates data is stored in ``shared_state``, which is also
responsible for managing the lifetime and initialisation of said data. This is
done to allow mechanisms to be implemented as essentially stateless bits of
C-code interacting with Arbor only through ``shared_state``. See the ABI
documentation for a discussion of the details.

Further Functionality
---------------------

In addition the configuration and effect of stimuli, gap junctions, and
sampling, as well as the computation of per-CV time steps is handled here.
