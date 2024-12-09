.. _cppremote:

Remote Communication
####################

Wraps remote communication for Arbor. This is meant to facilitate sending data
_to_ Arbor, not for pulling data into Arbor from the outside, which is done
automatically. If you are developing a bridge between Arbor and another
simulator that supports calling C++, this is the correct place. In all other
cases it is likely not what you are looking for. For a description of the
protocol see :ref:`here <interconnectivity>`

.. cpp:namespace:: arb::remote

Control Messages
================

.. cpp:class:: msg_null

    Empty message, possibly used as a keep-alive signal.


.. cpp:class:: msg_abort

    Request termination, giving the reason as a message.

    .. cpp:member:: char reason[512]

.. cpp:class:: msg_epoch

    Commence next epoch, giving the open interval :math:`[from, to)` with times
    in `ms`.

    .. cpp:member:: double t_start

    .. cpp:member:: double t_end

.. cpp:class:: msg_done

    Conclude simulation, giving the final time :math:`t_{\mathrm{final}}` in `ms`.

    .. cpp:member:: double time

.. cpp:type:: ctrl_message =  std::variant<msg_null, msg_abort, msg_epoch, msg_done>

.. function:: exchange_ctrl(ctrl_message message, MPI_Comm comm)

    Send ``message`` to all peers in the MPI intercomm ``comm`` and receive the
    unanimous answer. ``message`` must be one of the types ``msg_*`` above.

Spike Exchange
==============

.. cpp:class:: arb_spike

    .. cpp:member:: uint32_t gid

        Global id of the spiking cell, must fit in an unsigned 32b integer.
        ``gid`` must be unique in the external network.

    .. cpp:member:: uint32_t lid

        Local id on the spiking cell, must fit in an unsigned 32b integer. This
        ``lid`` describes which logical part of the cell ``gid`` emitted the
        spike. If the external simulation doesn't distinguish betwenn different
        sources on the same cell, always set this to zero.

     .. cpp:member:: double time

        Time at which the event occured.

 .. function:: gather_spikes(const std::vector<arb_spike>& spikes, MPI_Comm comm)

        Sends a buffer of spikes over ``comm`` receiving back the concatenated
        result of all calling MPI tasks in Arbor. This is a collective
        operation; each MPI task on the remote side must call it simultaneously
        with its _local_ part of the spikes to send.
