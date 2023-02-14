#pragma once

#include <optional>
#include <string_view>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <array>
#include <stdexcept>

#include <arbor/export.hpp>

namespace arb {

// NOTE: Cannot use arbexcept since it include common_type, which includes this. Circular include!
struct ARB_SYMBOL_VISIBLE serdes_error: std::runtime_error {
    serdes_error(std::string_view w): std::runtime_error{std::string{w}} {}
};

struct ARB_SYMBOL_VISIBLE illegal_key_type: serdes_error {
    illegal_key_type(): serdes_error{"SerDes keys must be an integral or string-like type."} {}
};

// Handling keys
using key_type = std::string;

template <typename K>
key_type to_key(K&& key) {
    using T = std::decay_t<std::remove_cv_t<std::remove_reference_t<K>>>;
    if constexpr (std::is_same_v<std::string, T>) {
        return key;
    }
    else if constexpr (std::is_same_v<char*, T>) {
        return std::string{key};
    }
    else {
        return std::to_string(key);
    }
}

template <typename K>
void from_key(K& key, const key_type& k) {
    using T = std::decay_t<std::remove_cv_t<std::remove_reference_t<K>>>;
    if constexpr (std::is_same_v<std::string, T>) {
        key = k;
    }
    else if constexpr (std::is_same_v<char*, T>) {
        key = std::string{k};
    }
    else if constexpr (std::is_integral_v<T>) {
        key = std::stoll(k);
    }
    else {
        throw illegal_key_type{};
    }
}

struct ARB_SYMBOL_VISIBLE null_error: serdes_error {
    template<typename K>
    null_error(const K& k): serdes_error{"Trying to deref a null pointer for key " + to_key(k)} {}
};

struct serializer {
    template <typename I>
    serializer(I& i): wrapped{std::make_unique<wrapper<I>>(i)} {}

    void begin_write_map(const key_type& k) { wrapped->begin_write_map(k); }
    void end_write_map() { wrapped->end_write_map(); }
    void begin_write_array(const key_type& k) { wrapped->begin_write_array(k); }
    void end_write_array() { wrapped->end_write_array(); }

    void begin_read_map(const key_type& k) { wrapped->begin_read_map(k); }
    void end_read_map() { wrapped->end_read_map(); }
    void begin_read_array(const key_type& k) { wrapped->begin_read_array(k); }
    void end_read_array() { wrapped->end_read_array(); }

    void write(const key_type& k, std::string v) { wrapped->write(k, v); }
    void write(const key_type& k, double v) { wrapped->write(k, v); }
    void write(const key_type& k, long long v) { wrapped->write(k, v); }
    void write(const key_type& k, unsigned long long v) { wrapped->write(k, v); };

    void read(const key_type& k, std::string& v) { wrapped->read(k, v); };
    void read(const key_type& k, long long& v) { wrapped->read(k, v); };
    void read(const key_type& k, unsigned long long& v) { wrapped->read(k, v); };
    void read(const key_type& k, double& v) { wrapped->read(k, v); };

    std::optional<key_type> next_key() { return wrapped->next_key(); }

private:
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

    template <typename I>
    struct wrapper: interface {
        wrapper(I& i): inner(i) {}
        I& inner;

        void write(const key_type& k, std::string v) override { inner.write(k, v); }
        void write(const key_type& k, double v) override { inner.write(k, v); }
        void write(const key_type& k, long long v) override { inner.write(k, v); }
        void write(const key_type& k, unsigned long long v) override { inner.write(k, v); };

        void read(const key_type& k, std::string& v) override { inner.read(k, v); };
        void read(const key_type& k, long long& v) override { inner.read(k, v); };
        void read(const key_type& k, unsigned long long& v) override { inner.read(k, v); };
        void read(const key_type& k, double& v) override { inner.read(k, v); };

        std::optional<key_type> next_key() override { return inner.next_key(); }

        void begin_write_map(const key_type& k) override { inner.begin_write_map(k); }
        void end_write_map() override { inner.end_write_map(); }

        void begin_write_array(const key_type& k) override { inner.begin_write_array(k); }
        void end_write_array() override { inner.end_write_array(); }

        void begin_read_map(const key_type& k) override { inner.begin_read_map(k); }
        void end_read_map() override { inner.end_read_map(); }

        void begin_read_array(const key_type& k) override { inner.begin_read_array(k); }
        void end_read_array() override { inner.end_read_array(); }

        virtual ~wrapper() = default;
    };

    std::unique_ptr<interface> wrapped;
};

// the actual interface
template<typename K>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, const std::string& v) {
    ser.write(to_key(k), v);
}

template<typename K>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, std::string_view v) {
    ser.write(to_key(k), std::string{v});
}

template<typename K>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, const char* v) {
    ser.write(to_key(k), std::string{v});
}

template<typename K, typename P>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, P* p) {
    if (!p) throw null_error{k};
    serialize(ser, to_key(k), *p);
}

template<typename K, typename P>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, const std::unique_ptr<P>& p) {
    if (!p) throw null_error{k};
    serialize(ser, k, *p);
}

template<typename K, typename P>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, const std::shared_ptr<P>& p) {
    if (!p) throw null_error{k};
    serialize(ser, k, *p);
}

template<typename K>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, long v) {
    ser.write(to_key(k), static_cast<long long>(v));
}

template<typename K>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, int v) {
    ser.write(to_key(k), static_cast<long long>(v));
}

template<typename K>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, unsigned v) {
    ser.write(to_key(k), static_cast<unsigned long long>(v));
}

template<typename K>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, unsigned long v) {
    ser.write(to_key(k), static_cast<unsigned long long>(v));
}

template<typename K>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, const float v) {
    ser.write(to_key(k), double{v});
}

template<typename K>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, const double v) {
    ser.write(to_key(k), v);
}

template<typename K>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, const bool v) {
    ser.write(to_key(k), static_cast<long long>(v));
}

template<typename K>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, const unsigned long long v) {
    ser.write(to_key(k), v);
}

template <typename K,
          typename Q,
          typename V>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, const std::unordered_map<Q, V>& v) {
    ser.begin_write_map(to_key(k));
    for (const auto& [q, w]: v) serialize(ser, q, w);
    ser.end_write_map();
}

template <typename K,
          typename Q,
          typename V>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, const std::map<Q, V>& v) {
    ser.begin_write_map(to_key(k));
    for (const auto& [q, w]: v) serialize(ser, q, w);
    ser.end_write_map();
}

template <typename K,
          typename V,
          typename A>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, const std::vector<V, A>& vs) {
    ser.begin_write_array(to_key(k));
    for (std::size_t ix = 0; ix < vs.size(); ++ix) serialize(ser, ix, vs[ix]);
    ser.end_write_array();
}

template <typename K,
          typename V,
          size_t N>
ARB_ARBOR_API void serialize(serializer& ser, const K& k, const std::array<V, N>& vs) {
    ser.begin_write_array(to_key(k));
    for (std::size_t ix = 0; ix < vs.size(); ++ix) serialize(ser, ix, vs[ix]);
    ser.end_write_array();
}

// Reading
template<typename K>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, std::string& v) {
    ser.read(to_key(k), v);
}

template<typename K, typename P>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, P* p) {
    if (!p) throw null_error{k};
    deserialize(ser, to_key(k), *p);
}

template<typename K, typename P>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, std::unique_ptr<P>& p) {
    if (!p) throw null_error{k};
    deserialize(ser, k, *p);
}

template<typename K, typename P>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, std::shared_ptr<P>& p) {
    if (!p) throw null_error{k};
    deserialize(ser, k, *p);
}

template<typename K>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, long& v) {
    long long tmp;
    ser.read(to_key(k), tmp);
    v = tmp;
}

template<typename K>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, int& v) {
    long long tmp;
    ser.read(to_key(k), tmp);
    v = tmp;
}

template<typename K>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, unsigned& v) {
    unsigned long long tmp;
    ser.read(to_key(k), tmp);
    v = tmp;
}

template<typename K>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, unsigned long& v) {
    unsigned long long tmp;
    ser.read(to_key(k), tmp);
    v = tmp;
}

template<typename K>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, unsigned long long& v) {
    ser.read(to_key(k), v);
}

template<typename K>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, float& v) {
    double tmp;
    ser.read(to_key(k), tmp);
    v = tmp;
}

template<typename K>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, double& v) {
    ser.read(to_key(k), v);
}

template<typename K>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, bool& v) {
    long long tmp;
    ser.read(to_key(k), tmp);
    v = tmp;
}

template <typename K,
          typename Q,
          typename V>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, std::unordered_map<Q, V>& vs) {
    ser.begin_read_map(to_key(k));
    for (;;) {
        auto q = ser.next_key();
        if (!q) break;
        typename std::remove_cv_t<Q> key;
        from_key(key, *q);
        if (!vs.count(key)) vs[key] = {}; // NOTE Must be default constructible anyhow
        deserialize(ser, *q, vs[key]);
    }
    ser.end_read_map();
}

template <typename K,
          typename Q,
          typename V>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, std::map<Q, V>& vs) {
    ser.begin_read_map(to_key(k));
    for (;;) {
        auto q = ser.next_key();
        if (!q) break;
        typename std::remove_cv_t<Q> key;
        from_key(key, *q);
        if (!vs.count(key)) vs[key] = {}; // NOTE Must be default constructible anyhow
        deserialize(ser, *q, vs[key]);
    }
    ser.end_read_map();
}

template <typename K,
          typename V,
          typename A>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, std::vector<V, A>& vs) {
    ser.begin_read_array(to_key(k));
    for (std::size_t ix = 0;; ++ix) {
        auto q = ser.next_key();
        if (!q) break;
        if (ix >= vs.size()) vs.emplace_back(); // NOTE Must be default constructible anyhow
        deserialize(ser, ix, vs[ix]);
    }
    ser.end_read_array();
}

template <typename K,
          typename V,
          size_t N>
ARB_ARBOR_API void deserialize(serializer& ser, const K& k, std::array<V, N>& vs) {
    ser.begin_read_array(to_key(k));
    for (std::size_t ix = 0; ix < vs.size(); ++ix) deserialize(ser, ix, vs[ix]);
    ser.end_read_array();
}

// Macros to (intrusively) (de)serialize a struct; use in the 'public' section
#define ARB_SERDES_EXPAND(x) x
#define ARB_SERDES_SELECT(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, NAME,...) NAME
#define ARB_SERDES_PUT(...)                                             \
    ARB_SERDES_EXPAND(ARB_SERDES_SELECT(__VA_ARGS__,                    \
                                        ARB_SERDES_PUT20,               \
                                        ARB_SERDES_PUT19,               \
                                        ARB_SERDES_PUT18,               \
                                        ARB_SERDES_PUT17,               \
                                        ARB_SERDES_PUT16,               \
                                        ARB_SERDES_PUT15,               \
                                        ARB_SERDES_PUT14,               \
                                        ARB_SERDES_PUT13,               \
                                        ARB_SERDES_PUT12,               \
                                        ARB_SERDES_PUT11,               \
                                        ARB_SERDES_PUT10,               \
                                        ARB_SERDES_PUT9,                \
                                        ARB_SERDES_PUT8,                \
                                        ARB_SERDES_PUT7,                \
                                        ARB_SERDES_PUT6,                \
                                        ARB_SERDES_PUT5,                \
                                        ARB_SERDES_PUT4,                \
                                        ARB_SERDES_PUT3,                \
                                        ARB_SERDES_PUT2,                \
                                        ARB_SERDES_PUT1)(__VA_ARGS__))
#define ARB_SERDES_PUT1(func)
#define ARB_SERDES_PUT2(func, v1) func(v1)
#define ARB_SERDES_PUT3(func, v1, v2) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT2(func, v2)
#define ARB_SERDES_PUT4(func, v1, v2, v3) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT3(func, v2, v3)
#define ARB_SERDES_PUT5(func, v1, v2, v3, v4) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT4(func, v2, v3, v4)
#define ARB_SERDES_PUT6(func, v1, v2, v3, v4, v5) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT5(func, v2, v3, v4, v5)
#define ARB_SERDES_PUT7(func, v1, v2, v3, v4, v5, v6) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT6(func, v2, v3, v4, v5, v6)
#define ARB_SERDES_PUT8(func, v1, v2, v3, v4, v5, v6, v7) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT7(func, v2, v3, v4, v5, v6, v7)
#define ARB_SERDES_PUT9(func, v1, v2, v3, v4, v5, v6, v7, v8) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT8(func, v2, v3, v4, v5, v6, v7, v8)
#define ARB_SERDES_PUT10(func, v1, v2, v3, v4, v5, v6, v7, v8, v9) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT9(func, v2, v3, v4, v5, v6, v7, v8, v9)
#define ARB_SERDES_PUT11(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT10(func, v2, v3, v4, v5, v6, v7, v8, v9, v10)
#define ARB_SERDES_PUT12(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT11(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)
#define ARB_SERDES_PUT13(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT12(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12)
#define ARB_SERDES_PUT14(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT13(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13)
#define ARB_SERDES_PUT15(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT14(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14)
#define ARB_SERDES_PUT16(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT15(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
#define ARB_SERDES_PUT17(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT16(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16)
#define ARB_SERDES_PUT18(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT17(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17)
#define ARB_SERDES_PUT19(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT18(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18)
#define ARB_SERDES_PUT20(func, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19) ARB_SERDES_PUT2(func, v1) ARB_SERDES_PUT19(func, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19)

#define ARB_SERDES_WRITE(v) serialize(ser, #v, t.v);
#define ARB_SERDES_READ(v) deserialize(ser, #v, t.v);

#define ARB_SERDES_ENABLE(T, ...)                                        \
    template <typename K>                                                \
    friend ARB_ARBOR_API void serialize(::arb::serializer& ser,          \
                      const K& k,                                        \
                      const T& t) {                                      \
        ser.begin_write_map(::arb::to_key(k));                           \
        ARB_SERDES_EXPAND(ARB_SERDES_PUT(ARB_SERDES_WRITE, __VA_ARGS__)) \
        ser.end_write_map();                                             \
    }                                                                    \
    template <typename K>                                                \
    friend ARB_ARBOR_API void deserialize(::arb::serializer& ser,        \
                            const K& k,                                  \
                            T& t) {                                      \
        ser.begin_read_map(::arb::to_key(k));                            \
        ARB_SERDES_EXPAND(ARB_SERDES_PUT(ARB_SERDES_READ, __VA_ARGS__))  \
        ser.end_read_map();                                              \
    }

#define ARB_SERDES_ENABLE_ENUM(T) \
    template <typename K>                                                \
    ARB_ARBOR_API void serialize(::arb::serializer& ser,                 \
                   const K& k,                                           \
                   const T& t) {                                         \
        serialize(ser, k, static_cast<long long>(t));                    \
    }                                                                    \
    template <typename K>                                                \
    ARB_ARBOR_API void deserialize(::arb::serializer& ser,               \
                     const K& k,                                         \
                     T& t) {                                             \
       long long tmp;                                                    \
       deserialize(ser, k, tmp);                                         \
       t = static_cast<T>(tmp);                                          \
    }
} // arb
