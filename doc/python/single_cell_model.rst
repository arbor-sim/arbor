.. _pysingle_cell_model:

Single cell models
==================

.. currentmodule:: arbor

.. py:class:: single_cell_model

   Wrapper for simplified description and execution of single cell models.
   Abstracts away the details of a :class:`recipe` for simulations of single,
   stand-alone cable cells.

    .. method:: single_cell_model(cable_cell)

       Construct a *single_cell_model* from a :class:`cable_cell`

    .. method:: run(tfinal, dt = 0.025)

       Run the model from time t=0 to t= ``tflinal`` with a dt= ``dt``.

    .. method:: probe(what, where, frequency)

       Sample a variable on the cell:

       :param what: Name of the variable to record (currently only 'voltage').
       :param where: :class:`location` at which to sample the variable.
       :param frequency: The frequency at which to sample [Hz].

    .. method:: spikes()

       Returns spike times [ms] after a call to ``run()``.

    .. method:: traces()

       Returns a list of :class:`trace` after a call to ``run()``. Each element in the
       list holds the trace of a probe on the cell.

    .. attribute:: properties

       The :class:`cable_global_properties` of the cell.

.. py:class:: trace

   Stores a trace obtained from a probe after running a model.

   .. attribute:: variable

      Name of the variable being recorded

   .. attribute:: loc

      :class:`location` of the trace

   .. attribute:: t

      Sample times [ms]

   .. attribute:: v

      Sample values [units specific to sample variable]

.. Note::

   The *single_cell_model* is used in our :ref:`quick start <gs_single_cell>` guide.
   The examples illustrate how to construct a :class:`cable_cell` and use it to form
   a :class:`single_cell_model`; how to add probes; how to run the model; and how to
   visualize the results.