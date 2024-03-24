.. _pylifcell:

LIF cells
===========

.. currentmodule:: arbor

.. py:class:: lif_cell

    A benchmarking cell (leaky integrate-and-fire), used by Arbor developers to test communication performance,
    with neuronal parameters:

    .. function:: lif_cell(source, target, **, tau_m, V_th, C_m, E_L, E_R, V_m, t_ref)

        Constructor: assigns the label ``source`` to the single built-in source
        on the cell; and assigns the label ``target`` to the single built-in
        target on the cell. Optional parameters may be given and set the
        correspoding physical parameters of the cell

    .. attribute:: source

        The label of the single built-in source on the cell. Used for forming
        connections from the cell in the :py:class:`arbor.recipe` by creating a
        :py:class:`arbor.connection`.

    .. attribute:: target

        The label of the single built-in target on the cell. Used for forming
        connections to the cell in the :py:class:`arbor.recipe` by creating a
        :py:class:`arbor.connection`.

    .. attribute:: tau_m

        Membrane potential decaying constant [ms].

    .. attribute:: V_th

        Firing threshold [mV].

    .. attribute:: C_m

        Membrane capacitance [pF].

    .. attribute:: E_L

        Resting potential [mV].

    .. attribute:: E_R

        Reset potential [mV].

    .. attribute:: V_m

        Initial value of the Membrane potential [mV].

    .. attribute:: t_ref

        Refractory period [ms].
