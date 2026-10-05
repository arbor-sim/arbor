.. _pyrecipe:

Recipes
=================

.. currentmodule:: arbor

The :class:`recipe` class documentation is below.

A recipe describes neuron models in a cell-oriented manner and supplies methods to provide cell information.
Details on why Arbor uses recipes and general best practices can be found in :ref:`modelrecipe`.

Recipe
------

.. class:: recipe

    Describe a model by describing the cells and network, without any information about how the model is to be represented or executed.

    All recipes derive from this abstract base class.
    
    Recipes provide a cell-centric interface for describing a model.
    This means that model properties, such as connections, are queried using the global identifier ``gid`` of a cell.
    In the description below, the term ``gid`` is used as shorthand for the cell with a global identifier.

    **Required Constructor**

    The constructor must be implemented and call the base class constructor, as at the moment, there is no way
    to instruct Python to do that automatically.
    
    .. note:: 
        Arbor's Python binding is a thin wrapper around the Arbor library, which is written in C++.
        Calling the base class constructor ensures correct initialization of memory in the underlying C++ class.
    
    A minimal constructor therefore looks like this:

    .. code-block:: python

        def __init__(self):
            arbor.recipe.__init__(self)

    **Required Member Functions**

    The following member functions (besides a constructor) must be implemented by every recipe:

    .. function:: num_cells()

        The number of cells in the model.

    .. function:: cell_kind(gid)

        The cell kind of the cell with global identifier ``gid`` (return type: :class:`arbor.cell_kind`).

    .. function:: cell_description(gid)

        A high-level description of the cell with global identifier ``gid``,
        for example, the morphology, synapses, and ion channels required to build a multi-compartment neuron.
        The type used to describe a cell depends on the kind of the cell.
        The interface for querying the kind and description of a cell is separate
        to allow the cell type to be provided without building a full cell description,
        which can be very expensive.

    **Optional Member Functions**

    .. function:: connections_on(gid)

        Returns a list of all the **incoming** connections to ``gid``. Each
        connection should have a valid synapse label ``connection.dest`` on the
        post-synaptic target ``gid``, and a valid source label
        ``connection.source.label`` on the pre-synaptic source
        ``connection.source.gid``. See :class:`cell_connection`.

        By default returns an empty list.

    .. function:: external_connections_on(gid)

        Returns a list of all the **incoming** connections to ``gid`` from a
        remote simulation. Each connection should have a valid synapse label
        ``connection.dest`` on the post-synaptic target ``gid``, and a valid
        source label ``connection.source.lid`` on the pre-synaptic source
        ``connection.source.gid``. See :class:`connection`. Note that in
        contrast to :class:`connection`, this uses two numeric identifiers
        for the source instead.

        By default returns an empty list.

    .. function:: network_description()

        Returns a network description, consisting of a network selection, network value for
        weight and delay, and a network dictionary.

        By default returns none.

    .. function:: gap_junctions_on(gid)

        Returns a list of all the gap junctions connected to ``gid``.
        Each gap junction ``gj`` should have a valid gap junction site label ``gj.local`` on ``gid``,
        and a valid gap junction site label ``gj.peer.label`` on ``gj.peer.gid``.
        See :class:`gap_junction_connection`.

        By default returns an empty list.

    .. function:: event_generators(gid)

        A list of all the :class:`event_generator` s that are attached to ``gid``.

        By default returns an empty list.

    .. function:: probes(gid)

        Returns a list specifying the probesets describing probes on the cell ``gid``.
        Each element in the list is an opaque object of type :class:`probe` produced by
        cell kind-specific probeset functions. Each probeset in the list
        has a corresponding probeset id of type :class:`cell_member`: an id ``(gid, i)``
        refers to the probes described by the ith entry in the list returned by ``get_probes(gid)``.

        By default returns an empty list.

    .. function:: global_properties(kind)

        The global properties of a model.

        This method needs to be implemented for :class:`arbor.cell_kind.cable`, where the
        properties include ion concentrations and reversal potentials; initial membrane voltage;
        temperature; axial resistivity; membrane capacitance; cv_policy; and a pointer
        to the mechanism catalogue. Also see :ref:`mechanisms_builtins`.

        By default returns an empty object.

    .. function:: cell_isometry(gid)

        Returns an isometry consisting of translation and rotation, which is applied to cell morphology to resolve global locations.

        By default, it returns an isometry without translation and rotation.

Cells
------

See :ref:`pycell`.

Synapses
--------

See :ref:`pyinterconnectivity`.

Event generator and schedules
-----------------------------

See :ref:`pygenerators`

Example
-------

Below is an example of a recipe construction of a ring network of multi-compartmental cells.
Because the interface for specifying cable morphology cells is under construction, the temporary
helpers in cell_parameters and make_cable_cell for building cells are used.

.. container:: example-code

    .. code-block:: python

        import sys
        import arbor as A

        class ring_recipe (A.recipe):

            def __init__(self, n=4):
                # The base C++ class constructor must be called first, to ensure that
                # all memory in the C++ class is initialized correctly.
                A.recipe.__init__(self)
                self.ncells = n
                self.params = A.cell_parameters()

            # The num_cells method that returns the total number of cells in the model
            # must be implemented.
            def num_cells(self):
                return self.ncells

            # The cell_description method returns a cell.
            def cell_description(self, gid):
                # Cell should have a synapse labeled "syn"
                # and a detector labeled "detector"
                return make_cable_cell(gid, self.params)

            # The kind method returns the type of cell with gid.
            # Note: this must agree with the type returned by cell_description.
            def cell_kind(self, gid):
                return A.cell_kind.cable

            # Make a ring network
            def connections_on(self, gid):
                src = (gid-1)%self.ncells
                w = 0.01
                d = 10 * U.ms
                return [A.connection((src,"detector"), "syn", w, d)]

            # Attach a generator to the first cell in the ring.
            def event_generators(self, gid):
                if gid==0:
                    sched = A.explicit_schedule([1*U.ms])
                    return [A.event_generator("syn", 0.1, sched)]
                return []

            def get_probes(self, id):
                # Probe just the membrane voltage at a location on the soma.
                return [A.cable_probe_membrane_voltage('(location 0 0)')]
