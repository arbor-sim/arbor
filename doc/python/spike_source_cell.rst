.. _pyspikecell:

Spike source cells
==================

.. currentmodule:: arbor

.. py:class:: spike_source_cell

    A spike source cell, that generates a user-defined sequence of spikes
    that act as inputs for other cells in the network.

    .. function:: spike_source_cell(source, schedule)

        Construct a spike source cell that generates spikes and give it the label ``source``. The label
        can be used for forming connections from the cell in the :py:class:`arbor.recipe` by creating a
        :py:class:`arbor.connection`


        - at regular intervals (using an :class:`arbor.regular_schedule`)
        - at a sequence of user-defined times (using an :class:`arbor.explicit_schedule`)
        - at times defined by a Poisson sequence (using an :class:`arbor.poisson_schedule`)

        :param source: label of the source on the cell.
        :param schedule: User-defined sequence of time points (choose from :class:`arbor.regular_schedule`, :class:`arbor.explicit_schedule`, or :class:`arbor.poisson_schedule`).
