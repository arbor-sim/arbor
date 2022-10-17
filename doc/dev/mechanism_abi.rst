.. _mechanism_abi:

Mechanism ABI
=============

Code describing membrane mechanisms can be implemented in 3 ways:

* custom C++/CUDA compiled into your Arbor program;
* NMODL, compiled through Arbor's ``modcc`` compiler;
* custom code exposed through the Arbor Mechanism Application Binary Interface (ABI).

Here you will find the information needed to connect Arbor to mechanism
implementations through the mechanism ABI. This ABI makes use of the
C type calling convection, so any language that can expose itself through a
`C FFI <https://en.wikipedia.org/wiki/Foreign_function_interface>`_
can be used to implement mechanisms. The advantage of this method is that you can write
and compile mechanisms in a multitude of languages, and store and distribute those mechanism libraries
in compiled and optimized form. Any Arbor application can use them without compilation, of course
assuming that both the mechanism and Arbor are compiled for the same target platform.

All functionality is offered via a single C header file in the Arbor include
directory, ``mechanism_abi.h``. The central datatypes here are
:c:struct:`arb_mechanism_type` and :c:struct:`arb_mechanism_interface`, laying out the metadata
and backend implementations respectively. A single :c:struct:`arb_mechanism_type`
instance may be used by multiple :c:struct:`arb_mechanism_interface` instances.

.. Note::
  Note that ``mechanism_abi.h`` is heavily commented and is useful as
  documentation in its own right, also when writing mechanisms in other languages than C(++).

Metadata: ``arb_mechanism_type``
--------------------------------

This type collects all information independent of the backend.

.. c:struct:: arb_mechanism_type

  This type collects all information independent of the backend.

  Metadata:

  .. c:member:: unsigned long abi_version

    mechanism was built using this ABI,
    should be ARB_MECH_ABI_VERSION

  .. c:member:: arb_mechanism_fingerprint fingerprint

    unique ID, currently ignored

  .. c:member:: const char*               name

    (catalogue-level) unique name

  .. c:member:: arb_mechanism_kind        kind

    one of:

    * point
    * density
    * reversal_potential
    * junction

  .. c:member:: bool                      is_linear

    synapses only: if the state G is governed by dG/dt = f(v, G, M(t)), where:
    M(t) =Σ wᵢδᵢ(t) weighted incoming events,
    then f is linear in G and M.

    If true, mechanisms must adhere to this contract.

    Ignored for everything else.

  .. c:member:: bool                      has_post_events

    implements post_event hook

  Tables:

  .. c:member:: arb_field_info*           globals
  .. c:member:: arb_size_type             n_globals
  .. c:member:: arb_field_info*           state_vars
  .. c:member:: arb_size_type             n_state_vars
  .. c:member:: arb_field_info*           parameters
  .. c:member:: arb_size_type             n_parameters
  .. c:member:: arb_ion_info*             ions
  .. c:member:: arb_size_type             n_ions

Tables
''''''

All tables are given as an integer size and an array. There are two
kinds of tables: :c:struct:`arb_field_info` and :c:struct:`arb_ion_info`.
:c:struct:`arb_field_info` holds parameters, state variables, and global constants;
:c:struct:`arb_ion_info` holds ion dependencies.

.. c:struct:: arb_field_info

  .. c:member:: const char* name

    Field name, can be used from Arbor to query/set field values.

  .. c:member:: const char* unit

    Physical units, just for introspection, not checked

  .. c:member:: arb_value_type default_value

    values will be initialised to this value

  .. c:member:: arb_value_type range_low

    valid range, lower bound, will be enforced

  .. c:member:: arb_value_type range_high

    valid range, upper bound, will be enforced

.. c:struct:: arb_ion_info

  .. c:member:: const char* name

    Ion name, eg Ca, K, ...

  .. c:member:: bool write_int_concentration

    writes Xi?

  .. c:member:: bool write_ext_concentration

    writes Xo?

  .. c:member:: bool write_rev_potential

    writes Er?

  .. c:member:: bool read_rev_potential

    uses Er?

  .. c:member:: bool read_valence

    Uses valence?

  .. c:member:: bool verify_valence

    Checks valence?

  .. c:member:: int expected_valence

    Expected value

Interlude: Parameter packs
--------------------------

In order to explain the interface type, we have to introduce
the type :c:struct:`arb_mechanism_ppack`. This record is used to pass all information to
and from the interface methods.

Objects of this type are always created and allocated by Arbor and passed
fully formed to the interface. At this point:

- Global data values are initialised
- pointers in ``ion_state_view`` are set to their associated values in shared
  state on the Arbor side
- pointers to state, parameters, globals, and constants are allocated and
  initialised to the given defaults.
- SIMD only: :c:member:`arb_mechanism_ppack.index_constraints` is set up.

.. c:struct:: arb_mechanism_ppack

  Global data:

  .. c:member:: arb_index_type width

    Number of CVs of this mechanism, size of arrays

  .. c:member:: arb_index_type n_detectors

    Number of threshold detectors

  .. c:member:: arb_index_type* vec_ci

    [Array] Map CV to cell

  .. c:member:: arb_index_type* vec_di

    [Array] Map

  .. c:member:: arb_value_type* vec_dt

    [Array] time step

  .. c:member:: arb_value_type* vec_v

    [Array] potential

  .. c:member:: arb_value_type* vec_i

    [Array] current

  .. c:member:: arb_value_type* vec_g

    [Array] conductance

  .. c:member:: arb_value_type* temperature_degC

    [Array] Temperature in celsius

  .. c:member:: arb_value_type* diam_um

    [Array] CV diameter

  .. c:member:: arb_value_type* time_since_spike

    Times since last spike; one entry per cell and detector.

  .. c:member:: arb_index_type* node_index

    Indices of CVs covered by this mechanism, size is width

  .. c:member:: arb_index_type* peer_index

    Indices of peer CV of each CV in ``node_index``, needed for gap-junction connections, size is width.

  .. c:member:: arb_index_type* multiplicity

    [Unused]

  .. c:member:: arb_value_type* weight

    [Array] Weight

  .. c:member:: arb_size_type mechanism_id

    Unique ID for this mechanism on this cell group

  .. c:member:: arb_deliverable_event_stream events

    Events during the last period

  .. c:member:: arb_constraint_partition     index_constraints

    Index restrictions, not initialised for all backends.

  User data:

  .. c:member:: arb_value_type** parameters

    [Array] setable parameters

  .. c:member:: arb_value_type** state_vars

    [Array] integrable state

  .. c:member:: arb_value_type*  globals

    global constant state

  .. c:member:: arb_ion_state*   ion_states

    [Array] views into shared state

Members tagged as ``[Array]`` represent one value per CV. To access the values
belonging to your mechanism, a level of indirection via :c:member:`arb_mechanism_ppack.node_index` is
needed.

.. admonition:: Example

  Let's assume mechanism ``hh`` is defined on two regions: ``R``
  comprising CVs ``0`` and ``1``, ``R'`` with a single CV ``9``. Then ``node_index
  = [0, 1, 9]`` and ``width = 3``. Arrays like ``vec_v`` will be of size ``3`` as
  well. To access the CVs' diameters, one would write:

  .. code:: c++

    for (auto cv = 0; cv < ppack.width; ++cv) {
        auto idx = node_index[cv];
        auto d   = ppack_um[idx];
    }

.. warning::
  Note that values in :c:member:`arb_mechanism_ppack.diam_um` cover _all_ CV's regardless whether they
  are covered by the current mechanisms. Reading or writing to those values
  is considered undefined behaviour. The same holds for all other fields of
  :c:struct:`arb_mechanism_ppack`.

User Data
'''''''''

This section is derived from the tables passed in via the metadata struct, see
above. One entry per relevant table entry is provided in the same order. So, if

.. code:: c

  arb_field_info globals[] = { arb_field_info { .name="A",
                                                .unit="lb ft / s",
                                                .default_value=42.0,
                                                .range_low=0,
                                                .range_high=123 },
                                arb_field_info { .name="B",
                                                .unit="kg m / s",
                                                .default_value=42.0,
                                                .range_low=0,
                                                .range_high=123 }};
  arb_mechanism_type m = { .n_globals=2, .globals=globals };

the ``globals`` field of the corresponding parameter pack would have two
entries, the first corresponding to ``A`` and initialised to 42.0 and the second
for ``B`` set to 42.0.

The evolution of the state variables is left to the implementation via
``integrate_state``, while ``globals`` and ``parameters`` are considered
read-only. The ion states internal concentration ``Xi``, external concentration
``Xo``, trans-membrane current ``iX`` may also be read and written. Note that
concurrent updates by multiple mechanisms might occur in any order and each
mechanism will only observe the initial values at the time step boundary. All
contribution by mechanisms are summed up into a final value. Further note that
accessing these values without declaring this via a relevant ``arb_ion_info`` in
the ``arb_mechanism_type`` is undefined behaviour. Parameter packs are specific
to a backend.

Implementation: ``arb_mechanism_interface``
-------------------------------------------

The interface methods will be called with allocated and initialised :c:struct:`arb_mechanism_ppack`
data. The actual layout is unspecified, but all pointers are allocated and set
by Arbor. This means that your code must read/write to these locations in memory,
and that you cannot change the pointer to point to another slice of memory allocated
by your code.

.. c:type:: void (*arb_mechanism_method)(arb_mechanism_ppack*);

This is the type of all interface methods. These are collected in the record
below with some metadata about the backend.

.. c:struct:: arb_mechanism_interface

  .. c:member:: arb_backend_kind   backend

    one of

    - cpu
    - gpu

  .. c:member:: arb_size_type      partition_width

    granularity for this backed, eg SIMD lanes

  Interface methods:

  .. c:member:: arb_mechanism_method init_mechanism

    - called once during instantiation,
    - setup initial state, corresponds to NMODL's INITIAL block,
    - will receive an allocated and initialised ppack object

  .. c:member:: arb_mechanism_method compute_currents

    - compute ionic currents and set them through pointers in `ion_state`, currents
      live in `current_density`
    - called during each integration time step
      - at the start for reversal potential mechanisms, *before* current reset
      - after event deliver for anything else

  .. c:member:: arb_mechanism_method apply_events

    This method is expected to consume a set of :c:struct:`arb_deliverable_event` and apply
    effects to internal state, found in :c:member:`arb_mechanism_ppack.events` which is of type
    :c:struct:`arb_deliverable_event_stream`.

    These structures are set up correctly externally, but are only valid during this call.
    The data is read-only for :c:member:`arb_mechanism_interface.apply_events`.

    - called during each integration time step, right after resetting currents
    - corresponding to ``NET_RECEIVE``

  .. c:member:: arb_mechanism_method advance_state

    - called during each integration time step, after solving Hines matrices
    - perform integration on state variables
    - state variables live in `state_vars`, with a layout described above

  .. c:member:: arb_mechanism_method write_ions

    - update ionic concentrations via the pointers in `ion_state`
    - called during each integration time step, after state integration

  .. c:member:: arb_mechanism_method post_event

    - used to implement spike time dependent plasticity
    - consumes :c:member:`arb_mechanism_ppack.time_since_spike`
    - called during each integration time step, after checking for spikes
    - if implementing this, also set :c:member:`arb_mechanism_type.has_post_events` to ``true`` in the metadata

.. c:struct:: arb_deliverable_event

  .. c:member::  arb_size_type   mech_id

    mechanism type identifier (per cell group).

  .. c:member::  arb_size_type   mech_index

    instance of the mechanism

  .. c:member::  arb_float_type  weight

    connection weight

.. c:struct:: arb_deliverable_event_stream

  .. c:member::  arb_size_type                n_streams

    number of streams

  .. c:member::  const arb_deliverable_event* events

    array of event data items

  .. c:member::  const arb_index_type*        begin

    array of offsets to beginning of marked events

  .. c:member::  const arb_index_type*        end

    array of offsets to end of marked events

SIMDization
-----------

If a mechanism interface processes arrays in SIMD bundles, it needs to set
:c:member:`arb_mechanism_interface.partition_width` to that bundle's width in units of ``arb_value_type``. The
library will set up :c:member:`arb_mechanism_ppack.index_constraints` in the
parameter pack. This structure describes which bundles can be loaded/stored as a
contiguous block, which ones must be gathered/scattered, which are to be
broadcast from a constant, and so on. The reason for this is the indirection via
:c:member:`arb_mechanism_ppack.node_index` mentioned before. Please refer to the documentation of our :ref:`SIMD
interface layer <simd>` for more information.

Making A Loadable Mechanism
---------------------------

Mechanisms interface with Arbor by providing a single function, returning
a structure

.. c:struct:: arb_mechanism

  .. c:member:: arb_get_mechanism_type type

                Pointer to a function ``arb_mechanism_type get_type()``

  .. c:member:: arb_get_mechanism_interface i_cpu

                Pointer to a function ``arb_mechanism_interface get_interface()``
                that returns a pointer to the CPU interface which may be
                ``null``.

  .. c:member:: arb_get_mechanism_interface i_gpu

                Pointer to a function ``arb_mechanism_interace get_interface()``
                that returns a pointer to the GPU interface which may be
                ``null``.

You can create mechanisms with both ``i_gpu`` and ``i_cpu`` returning ``null``,
but at least one of the interfaces must be provided or Arbor will refuse to load
the catalogue this mechanism.

The naming scheme is shown in the example below

.. code:: C

  arb_mechanism make_arb_default_catalogue_pas();

Writing Mechanisms Directly Against the ABI
-------------------------------------------
.. _abi_raw:

.. warning::

   This is a measure of last resort. Usage is not recommended unless you have a
   dire need and a solid understanding of Arbor, its internals, and the ABI.

If your use case requires features not supported in NMODL, you can write mechanisms
in C++ directly. See the ABI documentation for what callbacks need to be filled in.
These mechanisms can be compiled with ``arbor-build-catalogue`` as well and must be
present in the same folder as the NMODL files. Example

.. code-block::

   $> ls mod
   A.mod
   B.hpp B_cpu.cpp B_gpu.cpp B_gpu.cu
   C.hpp C_cpu.cpp C_gpu.cpp C_gpu.cu
   $> arbor-build-catalogue my mod --raw B C
   Building catalogue 'my' from mechanisms in 'mod'
   * NMODL
     * A
   * Raw
     * B
     * C

The ``--raw`` flag must go last due to the argument parsing.

For this to compile, the following must be upheld:

- For each mechanism ``M`` these files must be present

  - ``M.hpp`` must define the mechanism metadata and declare the used interfaces.
  - ``M_cpu.cpp`` must define the CPU interface. (You can disable this for raw
    mechanisms by passing ``-C``.)

  - If GPU support is used

    - ``M_gpu.cpp`` must define the GPU interface.
    - ``M_gpu.cu``  must define CUDA kernels.

- The interface names must adhere to the chosen catalogue name, eg here ``make_arb_my_catalogue_B_interface_multicore();``

  - names may only contain alphanumeric characters and underscores.
  - names must not contain multiple successive underscores.
  - in general, a valid ``C++`` variable name should be used.
