.. _pyinterconnectivity:

Interconnectivity
#################

.. currentmodule:: arbor

.. class:: connection

    Describes a connection between two cells, defined by source and destination end points (that is pre-synaptic and
    post-synaptic respectively), a connection weight and a delay time.

    The :attr:`dest` does not include the gid of a cell, this is because a :class:`arbor.connection` is bound to the
    destination cell which means that the gid is implicitly known.

    .. function:: connection(source, destination, weight, delay)

        Construct a connection between the :attr:`source` and the :attr:`dest` with a :attr:`weight` and :attr:`delay`.

    .. attribute:: source

        The source end point of the connection (type: :class:`arbor.cell_global_label`, which can be initialized with a
        (gid, label) or a (gid, (label, policy)) tuple. If the policy is not indicated, the default
        :attr:`arbor.selection_policy.univalent` is used).

    .. attribute:: dest

        The destination end point of the connection (type: :class:`arbor.cell_local_label` representing the label of the
        destination on the cell, which can be initialized with just a label, in which case the default
        :attr:`arbor.selection_policy.univalent` is used, or a (label, policy) tuple). The gid of the cell is
        implicitly known.

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

            def connections_on(gid):
               # construct a connection from the "detector" source label on cell 2
               # to the "syn" target label on cell gid with weight 0.01 and delay of 10 ms.
               src  = arbor.cell_global_label(2, "detector")
               dest = arbor.cell_local_label("syn") # gid of the destination is is determined by the argument to `connections_on`
               w    = 0.01
               d    = 10
               return [arbor.connection(src, dest, w, d)]

.. class:: gap_junction_connection

    Describes a gap junction between two gap junction sites.

    The :attr:`local` site does not include the gid of a cell, this is because a :class:`arbor.gap_junction_connection`
    is bound to the destination cell which means that the gid is implicitly known.

    .. note::

       A bidirectional gap-junction between two cells ``c0`` and ``c1`` requires two
       :class:`gap_junction_connection` objects to be constructed: one where ``c0`` is the
       :attr:`local` site, and ``c1`` is the :attr:`peer` site; and another where ``c1`` is the
       :attr:`local` site, and ``c0`` is the :attr:`peer` site. If :attr:`ggap` is equal
       in both connections, a symmetric gap-junction is formed, other wise the gap-junction is asymmetric.

    .. function::gap_junction_connection(peer, local, ggap)

        Construct a gap junction connection between :attr:`peer` and :attr:`local` with conductance :attr:`ggap`.

    .. attribute:: peer

        The gap junction site: the remote half of the gap junction connection (type: :class:`arbor.cell_global_label`,
        which can be initialized with a (gid, label) or a (gid, label, policy) tuple. If the policy is not indicated,
        the default :attr:`arbor.selection_policy.univalent` is used).

    .. attribute:: local

        The gap junction site: the local half of the gap junction connection (type: :class:`arbor.cell_local_label`
        representing the label of the destination on the cell, which can be initialized with just a label, in which case
        the default :attr:`arbor.selection_policy.univalent` is used, or a (label, policy) tuple). The gid of the
        cell is implicitly known.

    .. attribute:: ggap

        The gap junction conductance [μS].

.. class:: spike_detector

    A spike detector, generates a spike when voltage crosses a threshold. Can be used as source endpoint for an
    :class:`arbor.connection`.

    .. attribute:: threshold

        Voltage threshold of spike detector [mV]

