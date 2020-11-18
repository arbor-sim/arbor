.. _nmodl:

NMODL
======

*NMODL* is a `DSL <https://www.neuron.yale.edu/neuron/static/py_doc/modelspec/programmatic/mechanisms/nmodl.html>`_
for describing ion channel and synapse dynamics that is used by NEURON,
which provides the mod2c compiler parses dynamics described in NMODL to
generate C code that is called from NEURON.

Arbor has an NMODL compiler, *modcc*, that generates
optimized code in C++ and CUDA, which is optimzed for
the target architecture. NMODL does not have a formal specification,
and its semantis are often
ambiguous. To manage this, Arbor uses its own dialect of NMODL that
does not allow some constructions used in NEURON's NMODL.

.. note::
    We hope to replace NMODL with a DSL that is well defined, and easier
    for both users and the Arbor developers to work with in the long term.
    Until then, please write issues on our GitHub with any questions
    that you have about getting your NMODL files to work in Arbor.

This page is a collection of NMODL rules for Arbor. It assumes that the reader
alreay has a working knowledge of NMODL.

Ions
-----

* Arbor recognizes ``na``, ``ca`` and ``k`` ions by default. Any new ions
  must be added explicitly in Arbor along with their default properties and
  valence (this can be done in the recipe or on a single cell model).
  Simply specifying them in NMODL will not work.
* The parameters and variabnles of each ion referenced in a ``USEION`` statement
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
  NMODL mechanism they need to be specified as ``STATE`` variables.

Special variables
-----------------

* Arbor exposes some parameters from the simulation to the NMODL mechanisms.
  These include ``v``, ``diam``, ``celsius`` in addition to the previously
  mentioned ion parameters.
* Special variables should not be ``ASSIGNED`` or ``CONSTANT``,
  they are ``PARAMETER``.
* ``diam`` and ``celsius`` can be set from the simulation side.
* ``v`` is a reserved varible name and can be written in NMODL.
* If Special variables are used in a ``PROCEDURE`` or ``FUNCTION``, they need
  to be passed as arguments.
* ``dt`` is not exposed to NMODL mechanisms.

Functions and blocks
---------------------

* ``SOLVE`` statements should be the first statement in the ``BREAKPOINT`` block.
* The return variable of ``FUNCTION`` has to always be set. ``if`` without associated
  ``else`` can break that if users are not careful.

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

