.. _pyinterconnectivity:

Interconnectivity
#################

.. class:: connection

    Describes a connection between two cells, defined by source and destination end points (that is pre-synaptic and post-synaptic respectively),
    a connection weight and a delay time.

    .. function:: connection(source, destination, weight, delay)

        Construct a connection between the :attr:`source` and the :attr:`dest` with a :attr:`weight` and :attr:`delay`.

    .. attribute:: source

        The source end point of the connection (type: :class:`arbor.cell_member`).

    .. attribute:: dest

        The destination end point of the connection (type: :class:`arbor.cell_member`).

    .. attribute:: weight

        The weight delivered to the target synapse.
        The weight is dimensionless, and its interpretation is specific to the type of the synapse target.
        For example, the expsyn synapse interprets it as a conductance with units μS (micro-Siemens).

    .. attribute:: delay

        The delay time of the connection [ms]. Must be positive.

    An example of a connection reads as follows:

    .. container:: example-code

        .. code-block:: python

            import arbor

            # construct a connection between cells (0,0) and (1,0) with weight 0.01 and delay of 10 ms.
            src  = arbor.cell_member(0,0)
            dest = arbor.cell_member(1,0)
            w    = 0.01
            d    = 10
            con  = arbor.connection(src, dest, w, d)

.. class:: gap_junction_connection

    Describes a gap junction between two gap junction sites.
    Gap junction sites are represented by :class:`arbor.cell_member`.

    .. function::gap_junction_connection(local, peer, ggap)

        Construct a gap junction connection between :attr:`local` and :attr:`peer` with conductance :attr:`ggap`.

    .. attribute:: local

        The gap junction site: one half of the gap junction connection.

    .. attribute:: peer

        The gap junction site: other half of the gap junction connection.

    .. attribute:: ggap

        The gap junction conductance [μS].
