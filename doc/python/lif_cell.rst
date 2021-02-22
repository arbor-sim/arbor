.. _pylifcell:

LIF cells
===========

.. currentmodule:: arbor

.. py:class:: lif_cell

    A benchmarking cell (leaky integrate-and-fire), used by Arbor developers to test communication performance,
    with neuronal parameters:

    .. attribute:: tau_m

        Membrane potential decaying constant [ms].

    .. attribute:: V_th

        Firing threshold [mV].

    .. attribute:: C_m

        Membrane capacitance [pF].

    .. attribute:: E_L

        Resting potential [mV].

    .. attribute:: V_m

        Initial value of the Membrane potential [mV].

    .. attribute:: t_ref

        Refractory period [ms].

    .. attribute:: V_reset

        Reset potential [mV].