.. _probesample:

Probing and Sampling
====================

Both cable cells and LIF cells can be probed, see here for more details on cells
:ref:`modelcells`. The LIF cell, however, has a much smaller set of observable
quantities and only offers scalar probes. Thus, the following discussion is
tailored to the cable cell.

Definitions
***********

.. glossary::

    probe
        A measurement that can be performed on a cell. Each cell kind will have its own sorts of probe.
        Cable cells (:py:attr:`arbor.cable_probe`) allow the monitoring of membrane voltage, total membrane
        current, mechanism state, and a number of other quantities, measured either over the whole cell,
        or at specific sites (see :ref:`pycablecell-probesample`).

    vector probe
        Certain probes work over a :term:`region` rather than a :term:`locset`. This means that, depending
        settings such as :term:`cv policy`, data is sampled as a function of distance, yielding multiple data points.
        Such probes are distinguished from regular probes as "vector probes".
        
    probeset
        A set of probes. Probes are placed on locsets, and therefore may describe more than one probe.

    probeset address
        Probesets are located at a probeset address, and the collection of probeset addresses for a given cell is
        provided by the :py:class:`recipe` object. One address may correspond to more than one probe:
        as an example, a request for membrane voltage on a cable cell at sites specified by a locset
        expression will generate one probe for each site in that locset expression.

        See :ref:`pycablecell-probesample` for a list of objects that return a probeset address.

    probeset id
        A designator for a probeset as specified by a recipe. The *probeset id* is a
        :py:class:`cell_member` referring to a specific cell by gid, and the index into the list of
        probeset addresses returned by the recipe for that gid.

    metadata
        Each probe has associated metadata describing, for example, the location on a cell where the
        measurement is being taken, or other such identifying information. Metadata for the probes
        associated with a :term:`probeset id` can be retrieved from the simulation object, and is also provided
        along with any recorded samples.

    sampler
    sample_recorder
        A sampler is something that receives :term:`probeset` data. It amounts to setting a particular probeset to a
        particular measuring schedule, and then having a :term:`handle` with which to access the recorded probeset data later on.

    sample
        A record of data corresponding to the value at a specific *probe* at a specific time.

    handle
        A handle for reaching sampling data associated to a sampler, which is associated to a probeset.
        Setting a sampler (through :py:func:`simulation.sample`) returns handles. Sampled data can be retrieved
        by passing the handle associated to a sampler (associated to a probeset) to :py:func:`simulation.samples`.

    schedule
        An object representing a series of monotonically increasing points in time, used for determining
        sample times (see :ref:`pyrecipe`).

Spiking
*******

Threshold detectors have a dual use: they can be used to record spike times, but are also used in propagating signals
between cells. See also :term:`threshold detector` and :ref:`cablecell-threshold-detectors`.

API
---

* :ref:`Python <pycablecell-probesample>`
* :ref:`C++ <cppcablecell-probesample>`
