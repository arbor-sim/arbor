.. _cable_cell:

How the Cable Cell is made
==========================

This is a short tour through the inner workings of a cable cell intended for new
Arbor developers. Cable cells are not the sole cell type supported in Arbor, but
both the most common and most complex kind.

This is the introduction from which more detailed descriptions will branch out.
As such, we will start with a simple cable cell simulation and how the user input
is turned into an executable object.

Terminology
-----------

In Arbor's codebase some prefixes are used as a low-key namespacing

- ``arb_``:: Mech ABI types, in general use through out Arbor, eg
  ``arb_mechanism_type``.
- ``fvm_``:: Concerning use by the Finite Volume Method (FVM), eg
  ``fvm_lowered_cell``.
- ``mc_``:: Related to Multi-Compartment (Cells), identical to cable cells the
  difference is purely historical, eg ``mc_cell_group``.

Setting up a Cable Cell simulation
----------------------------------

Arbor constructs a runnable simulation from three ingredients:

- ``recipe``:: a queryable structure that collects cells, connections, gap
  junctions, etc.
- ``context``:: a hardware configuration, summarising threads, GPU, and MPI
  resources to be used.
- ``domain_decomposition``:: Distribution of cells over the ``context``, made
  from ``context`` and ``recipe``.

The interesting part here is the ``recipe``, which is used to lazily produce the
data required by the ``simulation`` and ``domain_decomposition``. A simple example
might be this model of a single cell

.. code:: c++

   struct recipe: public arb::recipe {
       recipe(const arb::morphology& m, const arb::decor& d): cell{m, d} {}

       arb::cell_size_type num_cells() const override { return 1; }

       std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override { return {}; }

       arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::cable; }

       std::any get_global_properties(arb::cell_kind) const override { return gprop; }

       arb::util::unique_any get_cell_description(arb::cell_gid_type) const override { return cell; }

       arb::cable_cell cell
       arb::cable_cell_global_properties gprop;
  };

As you can see, data is produced on request by feeding the recipe a
``cell_gid_type``. Finally, we need to have a ``morphology`` and a ``decor`` to
generate a ``cable_cell``. Please refer to the documentation on how to construct
these objects. For now, it is sufficient to note that a ``cable_cell`` is a
description of a cell, consisting of a cell geometry and a mapping of
sub-geometries to properties, rather an object that can be simulated. At this point
ion channels are described by a structure ``(name, [(parameter, value)])``.

Lowered Cells, Shared State, and the Discretisation
---------------------------------------------------

To obtain a simulation we need to turn the ``cable_cell`` description object
into a ``fvm_lowered_cell``. However, multiple cells are collected into a
``mc_cell_group`` and ``fvm_lowered_cell`` is the lowered representation of a
full cell group. The ``fvm_lowered_cell`` is responsible for holding the
backend-specific data of a cell group, managing sampling and stimuli, facilitate
event processing, and driving time integration.

Discretisation splits the segments described by the morphology into *control
volumes* (CV; sometimes called *compartments*) according to a ``cv_policy``.
This allows us to construct a system of linear equations, the Hines matrix, to
describe the evolution of the CV voltages according to the cable equation. Refer
to :ref:`Discretisation <discretisation>` and :ref:`Cable equation
<cable_equation>`.

Backend-dependent data is stored in ``shared_state`` as per-compartment data and
indices into these arrays. Upon ``fvm_lowered_cell::initialize`` these are
populated using the concrete discretisation and the ``cable_cell`` description.
Also, mechanisms are concretised using the provided ``mechanism_catalogue`` and
their internal data is set up in ``shared_state``. See :ref:`Shared state <shared_state>`
for more details

Main integration loop
---------------------

Now we are in a state to simulate a cell group by calling
``simulation::run(t_end, dt)``.

The integration in Arbor proceeds in *epochs* with a length less than half a
time constant :math:`T_{min}`, which is the maximum time over which cell groups
can evolve independently. The length :math:`T_{min}` is derived as the minimum over all
axonal/synaptic delays. This works since we know that an event at time :math:`t`
can have an effect at time :math:`t + T_{min}` at the soonest. The factor of one
half stems from double-buffering to overlap communication and computation. So,
Arbor collects all events in an epoch and transmits them in bulk, see
:ref:`Communication <communication>` for details.

Integration in Arbor is then split into three parts:

1. apply effect of events to mechanisms :ref:`Event Handling <event_distribution>`
2. evolve mechanisms and apply currents :ref:`Mechanisms <mechanisms>`
3. solve voltage equations, see :ref:`Solver <matrix_solver>`

Integration proceeds as far as possible without needing to process an event, but
at most with the given time step `dt`.
