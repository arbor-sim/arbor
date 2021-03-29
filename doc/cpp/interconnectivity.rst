.. _cppinterconnectivity:

Interconnectivity
#################

.. cpp:class:: cell_connection

    Describes a connection between two cells: a pre-synaptic source and a
    post-synaptic destination. The source is typically a threshold detector on
    a cell or a spike source. The destination is a synapse on the post-synaptic cell.

    The :cpp:member:`dest` does not include the gid of a cell, this is because a
    :cpp:class:`cell_connection` is bound to the destination cell which means that the gid
    is implicitly known.

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

    Describes a gap junction between two gap junction sites. The :cpp:member:`local` site does not include
    the gid of a cell, this is because a :cpp:class:`gap_junction_connection` is bound to the local
    cell which means that the gid is implicitly known.

    .. note::

       A bidirectional gap-junction between two cells ``c0`` and ``c1`` requires two
       :cpp:class:`gap_junction_connection` objects to be constructed: one where ``c0`` is the
       :cpp:member:`local` site, and ``c1`` is the :cpp:member:`peer` site; and another where ``c1`` is the
       :cpp:member:`local` site, and ``c0`` is the :cpp:member:`peer` site. If :cpp:member:`ggap` is equal
       in both connections, a symmetric gap-junction is formed, other wise the gap-junction is asymmetric.

    .. cpp:member:: cell_member_type peer

        Peer gap junction site, represented by the pair (cell gid, gap junction site index on the cell)

    .. cpp:member:: cell_lid_type local

        Local gap junction site index on the cell, the gid of the local site's cell is implicitly known.

    .. cpp:member:: float ggap

        gap junction conductance in μS.
