.. _cppinterconnectivity:

Interconnectivity
#################

.. cpp:class:: cell_connection

    Describes a connection between two cells: a pre-synaptic source and a
    post-synaptic destination. The source is typically a threshold detector on
    a cell or a spike source. The destination is a synapse on the post-synaptic cell.

    .. cpp:type:: cell_connection_endpoint = cell_member_type

        Connection end-points are represented by pairs
        (cell index, source/target index on cell).

    .. cpp:member:: cell_connection_endpoint source

        Source end point.

    .. cpp:member:: cell_connection_endpoint dest

        Destination end point.

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
    Gap junction sites are represented by :cpp:type:cell_member_type.

    .. cpp:member:: cell_member_type local

        gap junction site: one half of the gap junction connection.

    .. cpp:member:: cell_member_type peer

        gap junction site: other half of the gap junction connection.

    .. cpp:member:: float ggap

        gap junction conductance in μS.
