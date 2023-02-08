.. _formatserdes:

Serialization/Deserialization (SerDes)
======================================

While not a fileformat on its own SerDes allows users to dump a snapshot of the
simulation state for a later time. This is **not** intended for data extraction,
but instead a reset the simulation state to this point in time. The saved data
is not a complete image of the state, but strives to be a minimal subset from
which to restore.

Usage
-----

SerDes in itself is independent of the file format used, it allows for multiple
backends. The general usage is show in the examples below; we assume familiarity
with the C++ and Python recipe interfaces:

  .. code:: c++

    // Storage
    auto writer = io{};
    auto serializer = arb::serdes::serializer{writer};

    // Construct a simulation.
    auto model = recipe{};
    auto ctx = arb::make_context();
    auto ddc = arb::partition_load_balance(model, ctx);
    auto simulation = arb::simulation{model, ddc, ctx};

    // Run forwards, snapshot state.
    simulation.run(T, dt);
    write(serializer, simulation);
    // Then run some more, ...
    simulation.run(2*T, dt);

    // ... rewind to time T...
    read(serializer, simulation);
    // ... and run the same segment again.
    simulation.run(2*T, dt);

    // At this point the results obtained in the two time segments
    // [T, 2T) should be identical.

Note that we left the definition of ``io`` open, as ``serializer`` uses it
through a well-defined interface. Thus, one can simply add new implementations.
Arbor ships with ``arborio::json_serdes`` that produces JSON output.

SerDes is currently not suitable to provide portable snapshots, and consequently
the following constraints apply:

1. The recipe class must be the same across serialize/deserialize
   Unfortunately, we cannot enforce this at runtime, since we lack the
   appropriate introspection capabilities in C++/Python.
2. Likewise, ``context`` and ``domain_decomposition`` must be identical. Again
   we cannot validate this, since we'd either stop migration between ranks or
   would have to instantiate and sort every local domain decomposition across
   all ranks.

Thus, the recommended approach is to treat a simulation 'script' (``.cxx`` /
``.py``), the parallel-runtime parameters (think ``mpirun``) and the associated
snapshots as a single unit.

Advanced: Writing your own (De)Serializers in C++
-------------------------------------------------

This is not available at the Python interface level, due to a mismatch in
features at the level of languages and binings generation.

All that is needed is to implement new overloads of the functions ``read`` and
``write``. For many C++ native types these exist, but some might be missing.
Likewise, your own class hierarchy might need serialization. For a given type
``T`` the signatures are

  .. code:: c++

    template<typename K>
    void write(serializer& ser, const K& k, const T& t);
    template<typename K>
    void read(serializer& ser, const K& k, const T& t);

and the key type ``K`` must be converted to the internal key type
``arb::serdes::key_type``. A convenience function ``key_type to_key(const K&)`` is
offered which works for integral and string types.

Array like value -- eg vectors and similar -- are stored like this

  .. code:: c++

    template <typename K,
              typename V,
              typename A>
    void write(serializer& ser, const K& k, const std::vector<V, A>& vs) {
        ser.begin_write_array(to_key(k));
        for (int ix = 0; ix < vs.size(); ++ix) write(ser, ix, vs[ix]);
        ser.end_write_array();
    }

and similar for map-like types

  .. code:: c++

    template <typename K,
              typename Q,
              typename V>
    void write(serializer& ser, const K& k, const std::map<Q, V>& v) {
        ser.begin_write_map(to_key(k));
        for (const auto& [q, w]: v) write(ser, q, w);
        ser.end_write_map();
    }

Reading data is a bit more involved, as writing data might be partial and work
only in conjunction with proper setup beforehand. Thus, one needs to take care
when overwriting values. The sotrage is polled for the next key using
``std::optional<key_type> next_key`` and the keys are converted using
``from_key`` to the native key type. Example

  .. code:: c++

    template <typename K,
              typename V,
              typename A>
    void read(serializer& ser, const K& k, std::vector<V, A>& vs) {
        ser.begin_read_array(to_key(k));
        for (int ix = 0;; ++ix) {
            auto q = ser.next_key();   // Poll next key
            if (!q) break;             // if nil, there's no more data in store.
            if (ix < vs.size()) {      // if the index is already present
                read(ser, ix, vs[ix]); // hand the value to `read` to be modified
            }
            else {                     // else create a new one.
                V val;
                read(ser, ix, val);
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

- Only store what is required. If values are constructed (correctly!)
  externally, don't store them, this will avoid problems and save space.
- Do no store data that might be required to change, example: anything related
  to data acquisition.
- When dealing with polymorphisim, add a trampoline like this

    .. code:: c++

        struct B {
            virtual void serialize(serdes::serializer& s, const std::string&) const = 0;
            virtual void deserialize(serdes::serializer& s, const std::string&) = 0;
        };

        void write(serdes::serializer& s, const std::string& k, const B& v) { v.serialize(s, k); }
        void read(serdes::serializer& s, const std::string& k, B& v) { v.deserialize(s, k); }

        struct D: B {
            ARB_SERDES_ENABLE(D, ...);

            virtual void serialize(serdes::serializer& s, const std::string&) const override { write(s, k, *this); };
            virtual void deserialize(serdes::serializer& s, const std::string&) override { read(s, k, *this); };
        };
