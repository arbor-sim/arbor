.. _pysinglecellmodel:

Single cell model
=================

.. currentmodule:: arbor

.. py:class:: single_cell_model

   Wrapper for simplified description and execution of single cell models.
   Only available in the python library.
   Abstracts away the details of a :class:`recipe`, :class:`context` and
   :class:`domain_decomposition` for simulations of single, stand-alone
   cable cells.
   The simulation can not be distributed over several machines.

   .. method:: single_cell_model(cable_cell)

      Construct a :class:`single_cell_model` from a :class:`cable_cell`

   .. method:: run(tfinal, dt)

      Run the model from time t= ``0`` to t= ``tflinal`` with a dt= ``dt``.

   .. method:: probe(what, where, frequency)

      Sample a variable on the cell:

      :param what: Name of the variable to record (currently only 'voltage').
      :param where: :class:`location` at which to sample the variable.
      :param frequency: The frequency at which to sample [kHz].

   .. method:: spikes()

      Returns a list spike times [ms] after a call to :class:`single_cell_model.run`.

   .. method:: traces()

      Returns a list of :class:`trace` after a call to  :class:`single_cell_model.run`.
      Each element in the list holds the trace of one of the probes added via
      :class:`single_cell_model.probe`.

   .. attribute:: properties

      The :class:`cable_global_properties` of the model.

   .. attribute:: catalogue

      The :class:`mechanism_catalogue` of the model.

.. py:class:: trace

   Stores a trace obtained from a probe after running a model.

   .. attribute:: variable

      Name of the variable being recorded. Currently only 'voltage'.

   .. attribute:: loc

      :class:`location` of the trace

   .. attribute:: t

      Sample times [ms]

   .. attribute:: v

      Sample values [units specific to sample variable]

.. Note::

   The :class:`single_cell_model` is used in our :ref:`tutorials <tutorialsinglecell>`.
   The examples illustrate how to construct a :class:`cable_cell` and use it to form
   a :class:`single_cell_model`; how to add probes; how to run the model; and how to
   visualize the results.