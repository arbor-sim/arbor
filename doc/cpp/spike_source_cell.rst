.. _cppspikecell:

Spike source cells
==================

.. cpp:namespace:: arb

.. cpp:class:: spike_source_cell

    A spike source cell, that generates a user-defined sequence of spikes
    that act as inputs for other cells in the network.

    .. function:: spike_source_cell(cell_tag_type source, const schedule& sched)

        Constructs a spike source cell. Spike source generate spikes and can be
        connected to using the label ``source`` in a cell in the
        :cpp:class:`recipe` by creating a :cpp:class:`cell_connection`

        :param source: label of the source on the cell.
        :param schedule: User-defined sequence of time points
