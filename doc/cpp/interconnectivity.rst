.. _cppinterconnectivity:

Interconnectivity
#################

.. cpp:class:: cell_connection

    Describes a connection between two cells: a pre-synaptic source and a
    post-synaptic destination. The source is typically a threshold detector on
    a cell or a spike source. The destination is a synapse on the post-synaptic cell.

    A :class:`cell_connection` is associated with the destination cell of the connection
    and its gid.

    .. cpp:member:: cell_member_type source

        Source end point, represented by the pair (cell gid, source index on the cell)

    .. cpp:member:: cell_lid_type dest

        Destination target index on the cell, target cell's gid is implicitly known.

    .. cpp:member:: float weight

        The weight delivered to the target synapse.
        The weight is dimensionless, and its interpretation is
        specific to the synapse type of the target. For example,
        the `expsyn` synapse interprets it as a conductance
        with units μS (micro-Siemens).

    .. cpp:member:: float delay

        Delay of the connection (milliseconds).

.. cpp:class:: gap_junction_connection

    Describes a gap junction between two gap junction sites.

    A :class:`gap_junction_connection` is associated with the local cell of the connection and
    its gid.

    .. cpp:member:: cell_member_type peer

        Peer gap junction site, represented by the pair (cell gid, gap junction site index on the cell)

    .. cpp:member:: cell_lid_type local

        Local gap junction site index on the cell, the gid of the local site's cell is implicitly known.

    .. cpp:member:: float ggap

        gap junction conductance in μS.
