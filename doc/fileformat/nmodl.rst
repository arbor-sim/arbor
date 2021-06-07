.. _formatnmodl:

NMODL
=====

.. csv-table::
   :header: "Name", "File extension", "Read", "Write"

   "NMODL", "``mod``", "✓", "✗"

*NMODL* is a `DSL <https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/mechanisms/nmodl.html>`_
for describing ion channel and synapse dynamics that is used by NEURON,
which provides the mod2c compiler parses dynamics described in NMODL to
generate C code that is called from NEURON.

Arbor has an NMODL compiler, *modcc*, that generates
optimized code in C++ and CUDA, which is optimized for
the target architecture. NMODL does not have a formal specification,
and its semantics are often
ambiguous. To manage this, Arbor uses its own dialect of NMODL that
does not allow some constructions used in NEURON's NMODL.

.. note::
    We hope to replace NMODL with a DSL that is well defined, and easier
    for both users and the Arbor developers to work with in the long term.
    Until then, please write issues on our GitHub with any questions
    that you have about getting your NMODL files to work in Arbor.

This page is a collection of NMODL rules for Arbor. It assumes that the reader
already has a working knowledge of NMODL.

Ions
-----

* Arbor recognizes ``na``, ``ca`` and ``k`` ions by default. Any new ions
  used in NMODL need to be explicitly added into Arbor along with their default
  properties and valence (this can be done in the recipe or on a single cell model).
  Simply specifying them in NMODL will not work.
* The parameters and variables of each ion referenced in a ``USEION`` statement
  are available automatically to the mechanism. The exposed variables are:
  internal concentration ``Xi``, external concentration ``Xo``, reversal potential
  ``eX`` and current ``iX``. It is an error to also mark these as
  ``PARAMETER``, ``ASSIGNED`` or ``CONSTANT``.
* ``READ`` and ``WRITE`` permissions of ``Xi``, ``Xo``, ``eX`` and ``iX`` can be set
  in NMODL in the ``NEURON`` block. If a parameter is writable it is automatically
  readable and doesn't need to be specified as both.
* If ``Xi``, ``Xo``, ``eX``, ``iX`` are used in a ``PROCEDURE`` or ``FUNCTION``,
  they need to be passed as arguments.
* If ``Xi`` or ``Xo`` (internal and external concentrations) are written in the
  NMODL mechanism they need to be declared as ``STATE`` variables and their initial
  values have to be set in the mechanism.

Special variables
-----------------

* Arbor exposes some parameters from the simulation to the NMODL mechanisms.
  These include ``v``, ``diam``, ``celsius`` and ``t`` in addition to the previously
  mentioned ion parameters.
* These special variables should not be ``ASSIGNED`` or ``CONSTANT``, they are
  ``PARAMETER``. This is different from NEURON where a built-in variable is
  declared ``ASSIGNED`` to make it accessible.
* ``diam`` and ``celsius`` are set from the simulation side.
* ``v`` is a reserved variable name and can be read but not written in NMODL.
* ``dt`` is not exposed to NMODL mechanisms.
* ``area`` is not exposed to NMODL mechanisms.
* ``NONSPECIFIC_CURRENTS`` should not be ``PARAMETER``, ``ASSIGNED`` or ``CONSTANT``.
  They just need to be declared in the NEURON block.

Functions, procedures and blocks
--------------------------------

* ``SOLVE`` statements should be the first statement in the ``BREAKPOINT`` block.
* The return variable of ``FUNCTION`` has to always be set. ``if`` without associated
  ``else`` can break that if users are not careful.
* Any non-``LOCAL`` variables used in a ``PROCEDURE`` or ``FUNCTION`` need to be passed
  as arguments.

Unsupported features
--------------------

* Unit conversion is not supported in Arbor (there is limited support for parsing
  units, which are just ignored).
* Unit declaration is not supported (ex: ``FARADAY = (faraday)  (10000 coulomb)``).
  They can be replaced by declaring them and setting their values in ``CONSTANT``.
* ``FROM`` - ``TO`` clamping of variables is not supported. The tokens are parsed and ignored.
  However, ``CONSERVE`` statements are supported.
* ``TABLE`` is not supported, calculations are exact.
* ``derivimplicit`` solving method is not supported, use ``cnexp`` instead.
* ``VERBATIM`` blocks are not supported.
* ``LOCAL`` variables outside blocks are not supported.
* ``INDEPENDENT`` variables are not supported.

Arbor-specific features
-----------------------

* Arbor's NMODL dialect supports the most widely used features of NEURON. It also
  has some features unavailable in NEURON such as the ``POST_EVENT`` procedure block.
  This procedure has a single argument representing the time since the last spike on
  the cell. In the event of multiple detectors on the cell, and multiple spikes on the
  detectors within the same integration period, the times of each of these spikes will
  be processed by the ``POST_EVENT`` block. Spikes are processed only once and then
  cleared.

  Example of a ``POST_EVENT`` procedure, where ``g`` is a ``STATE`` parameter representing
  the conductance:

  .. code::

    POST_EVENT(t) {
       g = g + (0.1*t)
    }
