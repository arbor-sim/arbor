.. _cppcablecell:

Cable cells
===========

.. toctree::
   :maxdepth: 1

   morphology
   probe_sample
   cable_cell_format

Cable cells, which use the :cpp:enum:`cell_kind` :cpp:expr:`cable`, represent
morphologically-detailed neurons as 1-d trees, with electrical and biophysical
properties mapped onto those trees.

A single cell is represented by an object of type :cpp:type:`cable_cell`.
Properties shared by all cable cells, as returned by the recipe
:cpp:expr:`get_global_properties` method, are described by an object of type
:cpp:type:`cable_cell_global_properties`.

The :cpp:type:`cable_cell` object
---------------------------------

Cable cells are minimally constructed from a :cpp:type:`morphology.` To add
dynamics (ion channels, synapses, ...) a :ref:`decor <cablecell-decoration>` may
be added and an :cpp:type:`label_dict` that associates names with particular
points (:cpp:type:`locset` objects) or subsets (:cpp:type:`region` objects) of
the morphology. The numerical discretization is declared using a
:cpp:type:`cv_policy`; if not given, the default will be used, see `here
<cablecell-discretization>`.

Morphologies are constructed from a :cpp:type:`segment_tree`, but can also be
generated via the :cpp:type:`stitch_builder`, which offers a slightly higher
level interface. Details are described in
:ref:`cppcablecell-morphology-construction`.

Each cell has particular values for its electrical and ionic properties. These
are determined first by the set of global defaults, then the defaults associated
with the cell, and finally by any values specified explicitly for a given
subsection of the morphology via the ``paint`` interface of the decor (see
:ref:`cppcablecell-electrical-properties` and
:ref:`cppcablecell-paint-properties`).

Ion channels and other distributed dynamical processes are also specified on the
decor via the ``paint`` method; while synapses, current clamps, gap junction
mechanisms, and the site for testing the threshold potential are specified via
the ``place`` method. See :ref:`cppcablecell-dynamics`, below.

.. _cppcablecell-dynamics:

Cell dynamics
-------------

Dynamics are imbued onto the cell by setting a :cpp:type:`decor` object during
construction. Each segment in a cell may have attached to it one or more density
*mechanisms*, which describe biophysical processes. These are processes that are
distributed in space, but whose behaviour is defined purely by the state of the
cell and the process at any given point.

Cells may also have *point* mechanisms, describing the dynamics at post-synaptic
sites. And *junction* mechanisms, describing the dynamics at each site of the
two sites of a gap-junction connection.

A fourth type of mechanism, which describes ionic reversal potential behaviour,
can be specified for cells or the whole model via cell parameter settings,
described below.

Mechanisms are described by a :cpp:type:`mechanism_desc` object. These specify
the name of the mechanism (used to find the mechanism in the mechanism
catalogue) and parameter values for the mechanism that apply within a segment. A
:cpp:type:`mechanism_desc` is effectively a wrapper around a name and a
dictionary of parameter/value settings.

.. cpp:class:: mechanism_desc

    .. cpp:function:: mechanism_desc(std::string name, std::vector<std::pair<std::string, double>> param = {})

        Construct from a name and an optional parameter list. Allows for
        implicit conversion.

    .. cpp:function:: mechanism_desc& set(const std::string& key, double value)

       Sets the parameter associated with :cpp:expr:`key` in the description.
       Returns a reference to the mechanism description, so that calls to
       :cpp:expr:`set` can be chained in a single expression.

    .. cpp:function:: double get(const std::string& key) const


:cpp:type:`density`, :cpp:type:`synapse` and :cpp:type:`junction` objects are
thin wrappers around a :cpp:type:`mechanism_desc`, needed for *painting* and
*placing* mechanisms on a :cpp:type:`decor`:

.. cpp:class:: density

   Construct a density wrapper from the mechanism `mech`.

.. cpp:class:: synapse

   Construct a synapse wrapper from the mechanism `mech`.

.. cpp:class:: junction

   Construct a junction wrapper from the mechanism `mech`.

The decor collects mechanisms and other settings

.. cpp:class:: decor

Density mechanisms are associated with a cable cell object with:

.. cpp:function:: void decor::paint(const region&, density)

Point mechanisms, which are associated with connection end points on a
cable cell, are placed on a set of locations given by a locset. The group
of generated items are given a label which can be used to create connections
in the recipe. Point mechanisms are attached to a cell with:

.. cpp:function:: void decor::place(const locset&, synapse, cell_tag_type label)

Gap-junction mechanisms, which are associated with gap-junction connection
end points on a cable cell, are placed on a single location given by a locset
(locsets with multiple locations will raise an exception). The generated item
is given a label which can be used to create gap-junction connections in the
recipe. Gap-junction mechanisms are attached to a cell with:

.. cpp:function:: void decor::place(const locset&, junction, cell_tag_type label)

Threshold detectors, serving as the source of connections and to record spikes
are attached to locsets like so

.. note::

   All mechanism types' parameters can be scaled using :ref:`inhomogeneous
   expressions <labels-iexpr>` and supplied as ``scaled_mechanism`` to ``place``
   and ``paint``.

.. cpp:function:: void decor::place(const locset&, threshold_detector, cell_tag_type label)

where

.. cpp:class:: threshold_detector

    Record a spike when the rising edge of the membrane potential crosses a
    limit.

    .. cpp:function:: threshold_detector(quantity limit)

        Construct from limit in [mV]

Finally, current stimuli are added to locations on the cell using

.. cpp:function:: void decor::place(const locset&, i_clamp)

where

.. cpp:class:: i_clamp

    .. cpp:class:: envelope_point

        Pair of ``time [ms]`` and ``amplitude [nA]``

        .. cpp:member:: double t
        .. cpp:member:: double amplitude

    .. cpp:function:: i_clamp()

       A default constructor, with an empty envelope, describes a trivial
       stimulus, providing no current at all.

    .. cpp:function:: i_clamp(const quantity& amplitude, const quantity& frequency=0*U::kHz, const quantity& phase=0*U::rad)

      Constant amplitude stimulus starting at t = 0.

      - **Amplitude** must be convertible to current
      - **Frequency**, must be convertible to frequency; gives a sine current if not zero
      - **Phase**, must be convertible to radians, phase shift of sine.

    .. cpp:function:: i_clamp(std::vector<envelope_point> envelope, const quantity& f = 0*U::kHz, const quantity& phi = 0*U::rad)

      Current described by piecewise linear amplitude from ``(time,
      amplitude)``. If ``frequency`` is non-zero, the sine's amplitude will be
      modulated by the envelope.

      - **Frequency**, must be convertible to frequency; gives a sine current if not zero
      - **Phase**, must be convertible to radians, phase shift of sine.

The current clamp provides a convenience constructor

.. cpp:function:: i_clamp::box(const quantity& onset, const quantity& duration, const quantity& amplitude, const quantity& frequency=0*kHz, const quantity& phase=0*rad)

   A 'box' stimulus with fixed onset time, duration, and constant amplitude.

Default values for the whole cell are set via

.. cpp:function:: decor& set_default(defaultable)

where ``defaultable`` is one of

- :cpp:expr:`init_membrane_potential` Initial membrane potential [mV]
- :cpp:expr:`temperature` Temperature [K]
- :cpp:expr:`axial_resistivity` Cable resistivity [Ω·cm]
- :cpp:expr:`membrane_capacitance` Membrane capacitance [F/m²]
- :cpp:expr:`init_int_concentration` Initial internal concentration [mM].
- :cpp:expr:`init_ext_concentration` Initial external concentration [mM].
- :cpp:expr:`ion_diffusivity` Diffusivity [m²/s]
- :cpp:expr:`init_reversal_potential` Initial reversal potential [mV].

see below for more details.

.. _cppcablecell-electrical-properties:

Electrical properties and ion values
-------------------------------------

The :cpp:type:`cable_cell_parameter_set` has the following members, where an
empty optional value or missing map key indicates that the corresponding value
should be taken from the cell or global parameter set.

.. cpp:class:: cable_cell_parameter_set

   .. cpp:member:: std::unordered_map<std::string, cable_cell_ion_data> ion_data

   The keys of this map are names of ions, whose parameters will be locally overridden.
   The struct :cpp:type:`cable_cell_ion_data` has three fields:
   :cpp:type:`init_int_concentration`, :cpp:type:`init_ext_concentration`, and
   :cpp:type:`init_reversal_potential`.

   Internal and external concentrations are given in millimolars, i.e. mol/m³.
   Reversal potential is given in millivolts.

   .. cpp:member:: util::optional<units::quantity> init_membrane_potential

   Initial membrane potential in millivolts.

   .. cpp:member:: util::optional<units::quantity> temperature

   Local temperature in Kelvin.

   .. cpp:member:: util::optional<units::quantity> axial_resistivity

   Local resistivity of the intracellular medium, in ohm-centimetres.

   .. cpp:member:: util::optional<units::quantity> membrane_capacitance

   Local areal capacitance of the cell membrane, in Farads per square metre.

   .. cpp:member:: util::optional<cv_policy> discretisation

   Method by which CV boundaries are determined when the cell is discretised.
   See :ref:`cv-policies`.

Default parameters for a cell are returned by the :cpp:expr:`default_parameters`
member in the :cpp:type:`cable_cell` object. This is a value of type
:cpp:type:`cable_cell_parameter_set`, which extends
:cpp:type:`cable_cell_parameter_set` by adding an additional field describing
reversal potential computation:

   .. cpp:member:: cable_cell_parameter_set::std::unordered_map<std::string, mechanism_desc> reversal_potential_method

   Maps the name of an ion to a 'reversal potential' mechanism that describes
   how it should be computed. When no mechanism is provided for an ionic
   reversal potential, the reversal potential will be kept at its initial value.

Default parameters for all cells are supplied in the
:cpp:type:`cable_cell_global_properties` struct, while per-cell defaults are set
via :cpp:expr:`decor::set_default`.

Global properties
-----------------

.. cpp:class:: cable_cell_global_properties

   .. cpp:member:: mechanism_catalogue catalogue

   all mechanism names refer to mechanism instances in this mechanism catalogue.
   by default, this is set to `global_default_catalogue()`, the
   catalogue that contains all mechanisms bundled with arbor.

   .. cpp:member:: std::optional<double> membrane_voltage_limit_mv

   if set, check to see if the membrane voltage ever exceeds this value in
   magnitude during the course of a simulation. if so, throw an exception and
   abort the simulation.

   .. cpp:member:: bool coalesce_synapses

   when synapse dynamics are sufficiently simple, the states of synapses within
   the same discretised element can be combined for better performance. this is
   true by default.

   .. cpp:member:: std::unordered_map<std::string, int> ion_species

   every ion species used by cable cells in the simulation must have an entry in
   this map, which takes an ion name to its charge, expressed as a multiple of
   the elementary charge. by default, it is set to include sodium ``na`` with
   charge 1, calcium ``ca`` with charge 2, and potassium ``k`` with charge 1.

   .. cpp:member:: cable_cell_parameter_set default_parameters

   the default electrical and physical properties associated with each cable
   cell, unless overridden locally. in the global properties, *every optional
   field must be given a value*, and every ion must have its default values set
   in :cpp:expr:`default_parameters.ion_data`.

   .. cpp:function:: add_ion(const std::string& ion_name, int charge, quantity init_iconc, quantity init_econc, quantity init_revpot)

   convenience function for adding a new ion to the global
   :cpp:expr:`ion_species` table, and setting up its default values in the
   `ion_data` table.

   .. cpp:function:: add_ion(const std::string& ion_name, int charge, quantity init_iconc, quantity init_econc, mechanism_desc revpot_mechanism)

   As above, but set the initial reversal potential to zero, and use the given
   mechanism for reversal potential calculation.

For convenience, :cpp:expr:`neuron_parameter_defaults` is a predefined
:cpp:type:`cable_cell_parameter_set` value that holds values that correspond to
NEURON defaults. To use these values, assign them to the
:cpp:expr:`default_parameters` field of the global properties object returned in
the recipe.

.. _cppcablecell-revpot:

Reversal potential dynamics
---------------------------

If no reversal potential mechanism is specified for an ion species, the initial
reversal potential values are maintained for the course of a simulation.
Otherwise, a provided mechanism does the work, but it subject to some strict
restrictions. A reversal potential mechanism described in NMODL:

* May not maintain any ``STATE`` variables.
* Can only write to the ``eX`` value associated with an ion ``X``.
* Can not be given as a ``POINT`` mechanism.

Essentially, reversal potential mechanisms must be pure functions of cellular
and ionic state.

If a reversal potential mechanism writes to multiple ions, then if the mechanism
is given for one of the ions in the global or per-cell parameters, it must be
given for all of them. Arbor's default catalogue includes a ``nernst`` reversal
potential, which is parameterized over a single ion, and so can be assigned to
e.g. calcium in the global parameters via

.. code:: c++

   cable_cell_global_properties gprop;
   // ...
   gprop.default_parameters.reversal_potential_method["ca"] = "nernst/ca";

This mechanism has global scalar parameters for the gas constant *R* and Faraday
constant *F*, corresponding to the exact values given by the 2019 redefinition
of the SI base units. These values can be changed in a derived mechanism in
order to use, for example, older values of these physical constants.

.. code:: c++

   gprop.catalogue.derive("nernst1998", "nernst", {{"R", 8.314472}, {"F", 96485.3415}});
   gprop.default_parameters.reversal_potential_method["ca"] = "nernst1998/ca";

.. _cppcablecell-paint-properties:

Overriding properties locally
-----------------------------

Physical properties can be set per :cpp:type:`region` using ``decor::paint``.
these are constructed using typed wrappers

.. cpp:class:: init_membrane_potential

    Initial membrane potential [mV].

.. cpp:class:: temperature

    Temperature [K]

.. cpp:class:: axial_resistivity

    Cable resistivity [Ω·cm]

.. cpp:class:: membrane_capacitance

    Membrane capacitance [F/m²]

Similarly, ionic properties can be set, they require the ion name as a string in
addition

.. cpp:class:: init_int_concentration

    Initial internal concentration [mM].

.. cpp:class:: init_ext_concentration

    Initial external concentration [mM].

.. cpp:class:: ion_diffusivity

    Diffusivity [m²/s]

.. cpp:class:: init_reversal_potential

        Initial reversal potential [mV].

.. note::

   All bio-phyiscale parameters can be scaled using :ref:`inhomogeneous
   expressions <labels-iexpr>` and supplied as an ``iexpr`` to ``paint``.
