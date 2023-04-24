.. _formatserdes:

Checkpointing
=============

While not a fileformat on its own, checkpoints allow users to dump a snapshot of
the simulation state for a later time. This is *not* intended for data
extraction (see :ref:`probesample` for this functionality), but instead a reset
the simulation state to this point in time. The saved data is not a complete
image of the state, but strives to be a minimal subset from which to restore
(see :ref:`below <impl guide>`).

Checkpoints are currently *not* portable; i.e. the following constraints apply:

1. The recipe class must be the same across ``serialize`` / ``deserialize``
   Unfortunately, we cannot enforce this at runtime, since we lack the
   appropriate introspection capabilities in C++/Python.
2. Likewise, ``context`` and ``domain_decomposition`` must be identical. Again
   we cannot validate this, since we'd either stop migration between ranks or
   would have to instantiate and sort every local domain decomposition across
   all ranks.
3. The hardware configuration must be identical: SIMD, GPU, and Scalar CPU code
   have different binary layouts which makes serializing across hardware settings
   invalid.

Thus, the recommended approach is to treat a simulation 'script' (``.cxx`` /
``.py``), the parallel-runtime parameters (think ``mpirun``) and the associated
snapshots as a single unit.

.. note:: While this functionality can quite obviously be abused to mutate certain
          variable out-of-band, we strongly advise against it.

Usage
-----

Checkpoints are independent of the storage engine used, it allows for multiple
backends. The general usage is show in the examples below; we assume familiarity
with the C++ and Python recipe interfaces:

  .. code:: c++

    // Storage engine, pluggable. See below.
    auto writer = io{};
    // Serializer object using the engine.
    auto serializer = arb::serializer{writer};

    // Construct a simulation.
    auto model = recipe{};
    auto ctx = arb::make_context();
    auto ddc = arb::partition_load_balance(model, ctx);
    auto simulation = arb::simulation{model, ddc, ctx};

    // Run forwards, snapshot state.
    simulation.run(T, dt);
    serialize(serializer, simulation);
    // Then run some more, ...
    simulation.run(2*T, dt);

    // ... rewind to time T...
    deserialize(serializer, simulation);
    // ... and run the same segment again.
    simulation.run(2*T, dt);

    // At this point the results obtained in the two time segments
    // [T, 2T) should be identical.

Note that we left the definition of ``io`` open, as ``serializer`` uses it
through a well-defined interface (see next section). Thus, one can simply add
new implementations. Arbor currently ships with ``arborio::json_serdes`` that
produces JSON output, more might be added over time.

In Python, currently only support for (de)serialization from/to JSON strings is
offered.

  .. code:: python

    import arbor as A

    rec = my_recipe()
    sim = A.simulation(rec)
    jsn = sim.serialize()
    sim.deserialize(jsn)


Writing your own Storage Engine (C++ only)
------------------------------------------

A storage engine is used to construct the serializer and is responsible for
writing out the data to whatever format and location required. Currently Arbor
offers a JSON engine in ``arborio/json_serdes.hpp`` which produces a JSON value
in memory. The serializer is polymorphic in the actual engine, which is only
require to implement the following interface.

   .. code:: c++

         struct interface {
             virtual void write(const key_type&, std::string) = 0;
             virtual void write(const key_type&, double) = 0;
             virtual void write(const key_type&, long long) = 0;
             virtual void write(const key_type&, unsigned long long) = 0;

             virtual void read(const key_type&, std::string&) = 0;
             virtual void read(const key_type&, double&) = 0;
             virtual void read(const key_type&, long long&) = 0;
             virtual void read(const key_type&, unsigned long long&) = 0;

             virtual std::optional<key_type> next_key() = 0;

             virtual void begin_write_map(const key_type&) = 0;
             virtual void end_write_map() = 0;
             virtual void begin_write_array(const key_type&) = 0;
             virtual void end_write_array() = 0;

             virtual void begin_read_map(const key_type&) = 0;
             virtual void end_read_map() = 0;
             virtual void begin_read_array(const key_type&) = 0;
             virtual void end_read_array() = 0;

             virtual ~interface() = default;
         };

The ``read`` and ``write`` methods are responsible for inserting and extracting
the relevant items. The ``begin_write_array`` and ``end_write_array`` methods
bracket a write of an array value and announce that the following keys are to
be interpreted as integer indices. Analogous for the ``map`` counterparts and
the associated ``begin_read`` and ``end_read`` methods. Finally, ``next_key`` is
used during reading of containers to retrieve an optional next key and advanced
the internal iterator. If empty, the container is exhausted, else the contained
key can be used to retrieve the associated value. See examples below and the JSON
interface in ``arborio``.


Adding Snapshotting to new Objects (C++ only)
---------------------------------------------

This is not available at the Python interface, due to a mismatch in features at
the level of languages and binings generation.

All that is needed is to implement new overloads of the functions ``read`` and
``write``. For many C++ native types these exist, but some might be missing.
Likewise, your own class hierarchy might need serialization. For a given type
``T`` the signatures are

  .. code:: c++

    template<typename K>
    void serialize(serializer& ser, const K& k, const T& t);
    template<typename K>
    void deserialize(serializer& ser, const K& k, const T& t);

and the key type ``K`` must be converted to the internal key type
``arb::key_type``. A convenience function ``key_type to_key(const K&)`` is
offered which works for integral and string types.

Array-like values -- eg vectors and similar -- are stored like this

  .. code:: c++

    template <typename K,
              typename V,
              typename A>
    void serialize(serializer& ser, const K& k, const std::vector<V, A>& vs) {
        ser.begin_write_array(to_key(k));
        for (std::size_t ix = 0; ix < vs.size(); ++ix) serialize(ser, ix, vs[ix]);
        ser.end_write_array();
    }

and similar for map-like types

  .. code:: c++

    template <typename K,
              typename Q,
              typename V>
    void serialize(serializer& ser, const K& k, const std::map<Q, V>& v) {
        ser.begin_write_map(to_key(k));
        for (const auto& [q, w]: v) serialize(ser, q, w);
        ser.end_write_map();
    }

Reading data is a bit more involved, as writing data might be partial and work
only in conjunction with proper setup beforehand. Thus, one needs to take care
when overwriting values. The storage is polled for the next key using
``std::optional<key_type> next_key`` and the keys are converted using
``from_key`` to the native key type. Example

  .. code:: c++

    template <typename K,
              typename V,
              typename A>
    void deserialize(serializer& ser, const K& k, std::vector<V, A>& vs) {
        ser.begin_read_array(to_key(k));
        for (std::size_t ix = 0;; ++ix) {
            auto q = ser.next_key();           // Poll next key
            if (!q) break;                     // if nil, there's no more data in store.
            if (ix < vs.size()) {              // if the index is already present
                deserialize(ser, ix, vs[ix]);  // hand the value to `read` to be modified
            }
            else {                             // else create a new one.
                V val;
                deserialize(ser, ix, val);
                vs.emplace_back(std::move(val));
            }
        }
        ser.end_read_array();
    }

For structures, use -- where possible -- the macro ``ARB_SERDES_ENABLE(type, field*)``
like this

   .. code:: c++

             struct T {
                std::string a;
                double b;
                std::vector<float> vs{1.0, 2.0, 3.0};

                ARB_SERDES_ENABLE(T, a, b, vs);
             };

which will define the required functions. Likewise ``enum (class)`` is treated with
``ARB_SERDES_ENABLE_ENUM``.

Guidelines
^^^^^^^^^^

.. _impl guide:

Only store mutable state required to reset to a given point. If values are
constructed externally, don't store them.

**Do not** store immutable or externally set items, that is

- global constants
- anything that will be constructed from the recipe: connections, cells, ...
- anything set by the user: samples, time step width, ...

**Do** store mutable state, like

- voltages, ion concentrations, current time, ... (``backends/*/shared_state.hpp``)
- mechanism state
- events in flight

When dealing with polymorphism, add a trampoline like this

    .. code:: c++

        struct B {
            virtual void serialize(serializer& s, const std::string&) const = 0;
            virtual void deserialize(serializer& s, const std::string&) = 0;
        };

        void serialize(serializer& s, const std::string& k, const B& v) { v.serialize(s, k); }
        void deserialize(serializer& s, const std::string& k, B& v) { v.deserialize(s, k); }

        struct D: B {
            ARB_SERDES_ENABLE(D, ...);

            virtual void serialize(serializer& s, const std::string&) const override { serialize(s, k, *this); };
            virtual void deserialize(serializer& s, const std::string&) override { deserialize(s, k, *this); };
        };
