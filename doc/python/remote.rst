.. _pyremote:

Remote Communication
####################

Wraps remote communication for Arbor. This is meant to facilitate sending data
_to_ Arbor, not for pulling data into Arbor from the outside, which is done
automatically. If you are developing a bridge between Arbor and another
simulator that is written in pure Python, this is the correct place. In all
other cases it is likely not what you are looking for. For a description of the
protocol see

.. currentmodule:: arbor

Control Messages
================

.. class:: msg_null

    Empty message, possibly used as a keep-alive signal.

    .. function:: msg_null()

.. class:: msg_abort

    Request termination, giving the reason as a message (< 512 bytes)

    .. function:: msg_abort(reason)

.. class:: msg_epoch

    Commence next epoch, giving the open interval :math:`[from, to)` with times
    in `ms`.

    .. function:: msg_epoch(from, to)

.. class:: msg_done

    Conclude simulation, giving the final time :math:`t_{\mathrm{final}}` in `ms`.

    .. function:: msg_done(tfinal)

.. function:: exchange_ctrl(message, comm)

    Send ``message`` to all peers in the MPI intercomm ``comm`` and receive the
    unanimous answer. ``message`` must be one of the types ``msg_*`` above.

Spike Exchange
==============

.. class:: arb_spike

    .. attribute:: gid

        Global id of the spiking cell, must fit in an unsigned 32b integer.
        ``gid`` must be unique in the external network.

    .. attribute:: lid

        Local id on the spiking cell, must fit in an unsigned 32b integer. This
        ``lid`` describes which logical part of the cell ``gid`` emitted the
        spike. If the external simulation doesn't distinguish betwenn different
        sources on the same cell, always set this to zero.

    .. attribute:: time

        Time at which the occured.

.. function:: gather_spikes(spikes, comm)

    Sends a buffer of spikes over ``comm`` receiving back the concatenated
    result of all calling MPI tasks in Arbor. This is a collective
    operation; each MPI task on the remote side must call it simultaneously
    with its _local_ part of the spikes to send.
