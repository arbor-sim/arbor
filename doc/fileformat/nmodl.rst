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

Units
-----

Arbor doesn't support unit conversion in NMODL. This table lists the key NMODL
quantities and their expected units.

===============================================  ===================================================  ==========
quantity                                         identifier                                           unit
===============================================  ===================================================  ==========
voltage                                          v / v_peer                                           mV
temperature                                      celsius                                              °C
diameter (cross-sectional)                       diam                                                 µm

current_density (density mechanisms)             identifier defined using ``NONSPECIFIC_CURRENT``     mA/cm²
conductivity (density mechanisms)                identifier inferred from current_density equation    S/cm²
                                                 e.g. in ``i = g*v`` g is the conductivity
current (point and junction mechanisms)          identifier defined using ``NONSPECIFIC_CURRENT``     nA
conductance (point and junction mechanisms)      identifier inferred from current equation            µS
                                                 e.g. in ``i = g*v`` g is the conductance
ion X current_density (density mechanisms)       iX                                                   mA/cm²

ion X current (point and junction mechanisms)    iX                                                   nA

ion X reversal potential                         eX                                                   mV
ion X internal concentration                     Xi                                                   mmol/L
ion X external concentration                     Xo                                                   mmol/L
ion X diffusive concentration                    Xd                                                   mmol/L
===============================================  ===================================================  ==========

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
  readable and must not be specified as both.
* If ``Xi``, ``Xo``, ``eX``, ``iX``, ``Xd`` are used in a ``PROCEDURE`` or ``FUNCTION``,
  they need to be passed as arguments.
* If ``Xi`` or ``Xo`` (internal and external concentrations) are written in the
  NMODL mechanism they need to be declared as ``STATE`` variables and their
  initial values have to be set in the ``INITIAL`` block in the mechanism. This
  transfers **all** responsibility for handling ``Xi`` / ``Xo`` to the mechanism
  and will lead to painted initial values to be ignored. If these quantities are
  not made ``STATE`` they may be written to, but their values will be reset to
  their initial values every time step.
* The diffusive concentration ``Xd`` does not share this semantics. It will not
  be reset, even if not in ``STATE``, and may freely be written. This comes at the
  cost of awkward treatment of ODEs for ``Xd``, see the included ``decay.mod`` for
  an example.
* ``Xd`` is present on all cables iff its associated diffusivity is set to a
  non-zero value.

Special variables
-----------------

* Arbor exposes some parameters from the simulation to the NMODL mechanisms.
  These include ``v``, ``diam``, and ``celsius`` in addition to the previously
  mentioned ion parameters.
* These special variables should not be ``ASSIGNED`` or ``CONSTANT``, they are
  ``PARAMETER``. This is different from NEURON where a built-in variable is
  declared ``ASSIGNED`` to make it accessible.
* ``diam`` and ``celsius`` are set from the simulation side.
* ``v`` is a reserved variable name and can be read but not written in NMODL.
* ``dt``, ``time``, and ``area`` are not exposed to NMODL mechanisms.
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
* ``FROM`` - ``TO`` clamping of variables is not supported. The tokens are
  parsed, and reported through the ``mechanism_info``, but otherwise ignored.
  However, ``CONSERVE`` statements are supported.
* ``TABLE`` is not supported, calculations are exact.
* ``derivimplicit`` solving method is not supported, use ``cnexp`` instead.
* ``VERBATIM`` blocks are not supported.
* ``LOCAL`` variables outside blocks are not supported.
* ``INDEPENDENT`` variables are not supported.

.. _arbornmodl:

Arbor-specific features
-----------------------

* It is required to explicitly pass 'magic' variables like ``v`` into procedures.
  It makes things more explicit by eliding shared and implicit global state. However, 
  this is only partially true, as having `PARAMETER v` brings it into scope, *but only* 
  in `BREAKPOINT`.
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

* Arbor allows a gap-junction mechanism to access the membrane potential at the peer site
  of a gap-junction connection as well as the local site. The peer membrane potential is
  made available through the ``v_peer`` variable while the local membrane potential
  is available through ``v``, as usual.
* Arbor offers a number of additional unary math functions which may offer improved performance
  compared to hand-rolled solutions (especially with the vectorized and GPU backends).
  All of the following functions take a single argument `x` and return a
  floating point value.

  ==================  ========================================  =========
  Function name       Description                               Semantics
  ==================  ========================================  =========
  sqrt(x)             square root                               :math:`\sqrt{x}`
  step_right(x)       right-continuous heaviside step           :math:`\begin{align*} 1 & ~~ \text{if} ~x \geq 0, \\ 0 & ~~ \text{otherwise}. \end{align*}`
  step_left(x)        left-continuous heaviside step            :math:`\begin{align*} 1 & ~~ \text{if} ~x \gt 0, \\ 0 & ~~ \text{otherwise}. \end{align*}`
  step(x)             heaviside step with half value            :math:`\begin{align*} 1 & ~~ \text{if} ~x \gt 0, \\ 0 & ~~ \text{if} ~x \lt 0, \\ 0.5 & ~~ \text{otherwise}. \end{align*}`
  signum(x)           sign of argument                          :math:`\begin{align*} +1 & ~~ \text{if} ~x \gt 0, \\ -1 & ~~ \text{if} ~x \lt 0, \\ 0 & ~~ \text{otherwise}. \end{align*}`
  exprelr(x)          smooth continuation over :math:`x=0` of   :math:`x/(1 - e^{-x})`
  ==================  ========================================  =========
  
Voltage Processes
-----------------

Some cases require direct manipulation of the membrane voltage ``v``; which is
normally prohibited and for good reason so. For these limited application,
however, we offer mechanisms that are similar to ``density`` mechanism, but are
tagged with ``VOLTAGE_PROCESS`` where normally ``SUFFIX`` would be used.

This is both a very sharp tool and a somewhat experimental feature. Depending on
our experience, it might be changed or removed. Using a ``VOLTAGE_PROCESS``,
voltage clamping and limiting can be implemented, c.f. relevant examples in the
``default`` catalogue. Example: limiting membrane voltage from above and below

.. code:: none

    NEURON {
        VOLTAGE_PROCESS v_limit
        GLOBAL v_low, v_high
    }

    PARAMETER {
        v_high =  20 (mV)
        v_low  = -70 (mV)
    }

    BREAKPOINT {
         v = max(min(v, v_high), v_low)
    }

As of the current implementation, we note the following details and constraints

* only the ``INITIAL`` and ``BREAKPOINT`` procedures are called.
* no ``WRITE`` access to ionic quantities is allowed.
* only one ``VOLTAGE_PROCESS`` maybe present on a single location, adding more
  results in an exception.
* the ``BREAKPOINT`` callback will execute _after_ the cable solver. A
  consequence of this is that if the initial membrane potential :math:`V_0` is
  unequal to that of a potentially applied voltage clamp :math:`V_c`, the first
  timestep will observe :math:`V_0`.

.. _format-sde:

Stochastic Processes
--------------------

Arbor supports :ref:`stochastic processes <mechanisms-sde>` in the form of stochastic differential
equations. The *white noise* sources can be defined in the model files using a ``WHITE_NOISE`` block:

.. code:: none

   WHITE_NOISE {
       a b 
       c
   }

Arbitrary white noise variables can be declared (``a, b, c`` in the example above). The
noise will be appropriately scaled with the numerical time step and can be considered unitless. In
order to influence the white noise generation, a seed value can be set at the level of the
simulation through the optional constructor argument ``seed``
(see :ref:`here <pysimulation>` or :ref:`here <cppsimulation>`).

If the state is updated by involving at least one of the declared white noise variables
the system is considered to be stochastic:

.. code:: none

   DERIVATIVE state {
       s' = f + g*a
   }

The solver method must then accordingly set to ``stochastic``:

.. code:: none

   BREAKPOINT {
       SOLVE state METHOD stochastic
   }

Nernst
------
Many mechanisms make use of the reversal potential of an ion (``eX`` for ion ``X``).
A popular equation for determining the reversal potential during the simulation is
the `Nernst equation <https://en.wikipedia.org/wiki/Nernst_equation>`_.
Both Arbor and NEURON make use of ``nernst``. Arbor implements it as a mechanism and
NEURON implements it as a built-in method. However, the conditions for using the
``nernst`` equation to change the reversal potential of an ion differ between the
two simulators.

1. In Arbor, the reversal potential of an ion remains equal to its initial value (which
has to be set by the user) over the entire course of the simulation, unless another
mechanism which alters that reversal potential (such as ``nernst``) is explicitly selected
for the entire cell. (see :ref:`cppcablecell-revpot` for details).

.. NOTE:
  This means that a user cannot indicate to use ``nernst`` to calculate the reversal
  potential on some regions of the cell, while other regions of the cell have a constant
  reversal potential. It's either applied on the entire cell or not at all. This differs
  from NEURON's policy.

2. In NEURON, there is a rule which is evaluated (under the hood) per section of a given
cell to determine whether or not the reversal potential of an ion remains constant or is
calculated using ``nernst``. The rule is documented
`here <https://neuron.yale.edu/neuron/static/new_doc/modelspec/programmatic/ions.html>`_
and can be summarized as follows:

  Examining all mechansims on a given section, if the internal or external concentration of
  an ion is **written**, and its reversal potential is **read but not written**, then the
  nernst equation is used **continuously** during the simulation to update the reversal
  potential of the ion.
  And if the internal or external concentration of an ion is **read**, and its reversal
  potential is **read but not written**, then the nernst equation is used **once** at the
  beginning of the simulation to caluclate the reversal potential of the ion, and then
  remains constant.
  Otherwise, the reversal potential is set by the user and remains constant.

One of the main consequences of this difference in behavior is that in Arbor, a mechanism
modifying the reversal potential (for example ``nernst``) can only be applied (for a given ion)
at a global level on a given cell. While in Neuron, different mechanisms can be used for
calculating the reversal potential of an ion on different parts of the morphology.
This is due to the different methods Arbor and NEURON use for discretising the morphology.
(A ``region`` in Arbor may include part of a CV, where as in NEURON, a ``section`` can only
contain full ``segments``).

Modelers are encouraged to verify the expected behavior of the reversal potentials of ions
as it can lead to vastly different model behavior.

Tips for Faster NMODL
---------------------

.. Note::
  If you are looking for help with NMODL in the context of NEURON this guide might not help.

NMODL is a language without formal specification and many unexpected
characteristics (many of which are not supported in Arbor), which results in
existing NMODL files being treated as difficult to understand and best left
as-is. This in turn leads to sub-optimal performance, especially since
mechanisms take up a large amount of the simulations' runtime budget. With some
understanding of the subject matter, however, it is quite straightforward to
obtain clean and performant NMODL files. We regularly have seen speed-ups
factors of roughly three from optimising NMODL.

First, let us discuss how NMODL becomes part of an Arbor simulation. NMODL
mechanisms are given in ``.mod`` files, whose layout and syntax has been
discussed above. These are compiled by ``modcc`` into a series of callbacks as
specified by the :ref:`mechanism_abi`. These operate on data held in Arbor's
internal storage. But, ``modcc`` does not generate machine code, it goes through
C++ (and/or CUDA) as an intermediary which is processed by a standard C++
compiler like GCC (or nvcc) to produce either a shared object (for external
catalogues) and code directly linked into Arbor (the built-in catalogues).

Now, we turn to a series of tips we found helpful in producing fast NMODL
mechanisms. In terms of performance of variable declaration, the hierarchy is
from slowest to fastest:

1. ``RANGE ASSIGNED`` -- mutable array
2. ``RANGE PARAMETER`` -- configurable array
3. ``ASSIGNED`` -- mutable
4. ``PARAMETER`` -- configurable
5. ``CONSTANT`` -- inlined constant


``RANGE``
~~~~~~~~~

Parameters and ``ASSIGNED`` variables marked as ``RANGE`` will be stored as an
array with one entry per CV in Arbor. Reading and writing these incurs a memory
access and thus affects cache and memory utilisation metrics. It is often more
efficient to use ``LOCAL`` variables instead, even if that means foregoing the
ability to re-use a computed value. Compute is so much faster than memory on
modern hardware that re-use at the expense of memory accesses is seldom
profitable, except for the most complex terms. ``LOCAL`` variables become just
that in the generated code: a local variable that is likely residing in a
register and used only as long as needed.

``PROCEDURE``
~~~~~~~~~~~~~

Prefer ``FUNCTION`` over ``PROCEDURE``. The latter *require* ``ASSIGNED RANGE``
variables to return values and thus stress the memory system, which, as noted
above, is not most efficient on current hardware. Also, they may not be inlined,
as opposed to a ``FUNCTION``.

``PARAMETER``
~~~~~~~~~~~~~

``PARAMETER`` should only be used for values that must be set by the simulator.
All fixed values should be ``CONSTANT`` instead. These will be inlined by
``modcc`` and propagated through the computations which can uncover more
optimisation potential.

Sharing Expressions Between ``INITIAL`` and ``BREAKPOINT`` or ``DERIVATIVE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is often done using a ``PROCEDURE``, which we now know is inefficient. On top,
this ``PROCEDURE`` will likely compute more outputs than strictly needed to
accomodate both blocks. DRY code is a good idea nevertheless, so use a series of
``FUNCTION`` instead to compute common expressions.

This leads naturally to a common optimisation in H-H style ion channels. If you
heeded the advice above, you will likely see this patter emerge:

.. code::

   na   = n_alpha()
   nb   = n_beta()
   ntau = 1/(na + nb)
   ninf = na*ntau

   n' = (ninf - n)/ntau

Written out in this explicit way it becomes obvious that this can be expressed
compactly as

.. code::

   na   = n_alpha()
   nb   = n_beta()
   nrho = na + nb

   n' = na - n*nrho

The latter code is faster, but neither ``modcc`` nor the external C++ compiler
will perform this optimisation [#]_. This is less easy to
see when partially hidden in a ``PROCEDURE``.

.. [#] GCC/Clang *might* attempt it if asked to relax floating point accuracy
       with ``-ffast-math`` or ``-Ofast``. However, Arbor refrains from using
       this option when compiling mechanism code.

Complex Expressions in Current Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``modcc``, Arbor's NMODL compiler, applies symbolic differentiation to the
current expression to find the conductance as ``g = d I/d U`` which are then
used to compute the voltage update. ``g`` is thus computed multiple times every
timestep and if the corresponding expression is inefficient, it will cost more
time than needed. The differentiation implementation quite naive and will not
optimise the resulting expressions. This is an internal detail of Arbor and
might change in the future, but for now this particular optimisation can help to
produce better performing code. Here is an example

.. code::

  : BAD, will compute m^4 * h every step
  i = m^4 * h * (v - e)

  : GOOD, will just use a constant value of g
  LOCAL g
  g = m^4 * h
  i = g * (v - e)

Note that we do not lose accuracy here, since Arbor does not support
higher-order ODEs and thus will treat ``g`` as a constant across
a single timestep even if ``g`` actually depends on ``v``.

Using Memory versus Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Commonly ion channels need to correct for temperature differences, which yields
a term similar to

.. code::

   q = 3^(0.1*celsius - 0.63)

Here, we find that the cost of the exponential when computing ``q`` in the
``DERIVATIVE`` block is high enough to make pre-computing ``q`` in ``INITIAL``
and loading the value later an optimisation. Shown below is a simplified version
of this pattern from ``hh.mod`` in the Arbor sources

.. code::

   NEURON {
     ...
     RANGE ..., q
   }

   ASSIGNED { q }

   PARAMETER {
       ...
       celsius (degC)
   }

   STATE { ... }

   BREAKPOINT {
       SOLVE dS METHOD cnexp
       ...
   }

   INITIAL {
      q = 3^(0.1*celsius - 0.63)
      ...
   }

   DERIVATIVE states {
      ... : uses q
   }

Specialised Functions
~~~~~~~~~~~~~~~~~~~~~

Some extra cost can be saved by choosing Arbor-specific optimized math functions instead of
hand-rolled versions. Please consult the table in :ref:`this section <arbornmodl>`.
A common pattern is the use of a guarded exponential of the form

.. code::

   if (x != 0) {
     r = a*x/(exp(-x) - 1)
   } else {
     r = a
   }

However, it can be written in Arbor's NMODL dialect as

.. code::

   exprelr(x)

which is more efficient and has the same guarantees. NMODL files originating
from NEURON often use this or related functions, e.g. ``vtrap(x, y) =
y*exprelr(x/y)``.

Small Tips and Micro-Optimisations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Divisions cost a bit more than multiplications and additions.
- ``m * m`` is more efficient than ``m^2``. This holds for higher powers as well
  and if you want to squeeze out the utmost of performance use
  exponentiation-by-squaring. (Although GCC does this for you. Most of the
  time.)
