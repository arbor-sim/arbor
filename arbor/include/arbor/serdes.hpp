#pragma once

#include <optional>
#include <string_view>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <array>
#include <type_traits>

#include <iostream>

namespace arb {
namespace serdes {

// type traits

// Have we got begin/end?
template <typename T, typename = void> struct is_iterable: std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())),
                                  decltype(std::end(std::declval<T>()))>>: std::true_type {};

template <typename T> constexpr bool is_iterable_v = is_iterable<T>::value;

// Found a key-value-like thingy?
template <typename T, typename = void> struct is_pair: std::false_type {};

template <typename T>
struct is_pair<T, std::void_t<decltype(std::get<0>(std::declval<T>())),
                              decltype(std::get<1>(std::declval<T>()))>>: std::true_type {};

template <typename T> constexpr bool is_pair_v = is_pair<T>::value;

// Anything we can deref?
template <typename T, typename = void> struct is_pointerlike: std::false_type {};

template <typename T>
struct is_pointerlike<T, std::void_t<decltype(*std::declval<T>())>>: std::true_type {};

template <typename T> constexpr bool is_pointerlike_v = is_pointerlike<T>::value;

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
        throw std::runtime_error{"Stupid key type"};
    }
}

struct serializer {
    template <typename I>
    serializer(I& i): wrapped{std::make_unique<wrapper<I>>(i)} {}

    template <typename K,
              typename V>
    void write(const K& key, const V& v) {
        auto k = to_key(key);
        using T = std::decay_t<V>;
        if constexpr ( std::is_same_v<char*, T>
                    || std::is_same_v<std::string, T>
                    || std::is_same_v<std::string_view, T>) {
            wrapped->write(k, std::string{v});
        }
        else if constexpr (is_iterable_v<T>) {
            if constexpr (is_pair_v<typename T::value_type>) {
                wrapped->begin_write_map(k);
                for (const auto& [q, w]: v) write(q, w);
                wrapped->end_write_map();
            }
            else  {
                int ix = 0;
                wrapped->begin_write_array(k);
                for (const auto& w: v) {
                    write(ix, w);
                    ix++;
                }
                wrapped->end_write_array();
            }
        }
        else if constexpr (is_pointerlike_v<T>) {
            write(k, *v);
        }
        else if constexpr (std::is_floating_point_v<T>) {
            wrapped->write(k, double{v});
        }
        else if constexpr (std::is_integral_v<T>) {
            wrapped->write(k, static_cast<long long>(v));
        }
        else {
            wrapped->begin_write_map(k);
            v.serialize(*this);
            wrapped->end_write_map();
        }
    }

    template <typename K, typename V, std::size_t N>
    void read(const K& key, std::array<V, N>& vs) {
        auto k = to_key(key);
        wrapped->begin_read_array(k);
        for (int ix = 0; ix < N; ++ix) {
            auto key = wrapped->next_key();
            if (!key) break;
            read(ix, vs[ix]);
        }
        wrapped->end_read_array();
    }

    template <typename K,
              typename V>
    void read(const K& key, V& v) {
        auto skey = to_key(key);
        using T = std::decay_t<V>;
        if constexpr (std::is_same_v<std::string, T>) {
            wrapped->read(skey, v);
        }
        else if constexpr (is_iterable_v<T>) {
            if constexpr (is_pair_v<typename T::value_type>) {
                wrapped->begin_read_map(skey);
                for (;;) {
                    auto q = wrapped->next_key();
                    if (!q) break;
                    std::remove_cv_t<typename T::value_type::first_type> k;
                    from_key(k, *q);
                    if (v.count(k)) {
                        read(k, v[k]);
                    }
                    else {
                        std::remove_cv_t<typename T::value_type::second_type> val;
                        read(k, val);
                        v[k] = std::move(val);
                    }
                }
                wrapped->end_read_map();
            }
            else {
                wrapped->begin_read_array(skey);
                for (int ix = 0;; ++ix) {
                    auto q = wrapped->next_key();
                    if (!q) break;
                    if (ix < v.size()) {
                        read(ix, v[ix]);
                    }
                    else {
                        typename V::value_type val;
                        read(ix, val);
                        v.emplace_back(std::move(val));
                    }
                }
                wrapped->end_read_array();
            }
        }
        else if constexpr (is_pointerlike_v<T>) {
            if (!v) throw std::runtime_error("Cannot deref a null at key " + skey);
            read(skey, *v);
        }
        else if constexpr (std::is_floating_point_v<T>) {
            double tmp;
            wrapped->read(skey, tmp);
            v = tmp;
        }
        else if constexpr (std::is_integral_v<T>) {
            long long tmp;
            wrapped->read(skey, tmp);
            v = tmp;
        }
        else {
            wrapped->begin_read_map(skey);
            v.deserialize(*this);
            wrapped->end_read_map();
        }
    }

    void begin_write_map(const key_type& k) { wrapped->begin_write_map(k); }
    void end_write_map() { wrapped->end_write_map(); }
    void begin_write_array(const key_type& k) { wrapped->begin_write_array(k); }
    void end_write_array() { wrapped->end_write_array(); }

    void begin_read_map(const key_type& k) { wrapped->begin_read_map(k); }
    void end_read_map() { wrapped->end_read_map(); }
    void begin_read_array(const key_type& k) { wrapped->begin_read_array(k); }
    void end_read_array() { wrapped->end_read_array(); }

private:
    struct interface {
        virtual void write(const key_type&, std::string) = 0;
        virtual void write(const key_type&, double) = 0;
        virtual void write(const key_type&, long long) = 0;

        virtual void read(const key_type&, std::string&) = 0;
        virtual void read(const key_type&, double&) = 0;
        virtual void read(const key_type&, long long&) = 0;

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

        void read(const key_type& k, std::string& v) override { inner.read(k, v); };
        void read(const key_type& k, long long& v) override { inner.read(k, v); };
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

// Macros to (intrusively) (de)serialize a struct; use in the 'public' section
//
#define ARB_SERDES_EXPAND(x) x
#define ARB_SERDES_SELECT(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, NAME,...) NAME
#define ARB_SERDES_PUT(...)                                             \
    ARB_SERDES_EXPAND(ARB_SERDES_SELECT(__VA_ARGS__,                    \
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

#define ARB_SERDES_WRITE(v) ser.write(#v, v);
#define ARB_SERDES_READ(v) ser.read(#v, v);

#define ARB_SERDES_ENABLE(...)                                          \
    void serialize(::arb::serdes::serializer& ser) const {              \
        ARB_SERDES_EXPAND(ARB_SERDES_PUT(ARB_SERDES_WRITE, __VA_ARGS__)) \
            }                                                           \
    void deserialize(::arb::serdes::serializer& ser) {                  \
        ARB_SERDES_EXPAND(ARB_SERDES_PUT(ARB_SERDES_READ, __VA_ARGS__)) \
            }

#define ARB_SERDES_FORWARD(ptr)                             \
    void serialize(::arb::serdes::serializer& ser) const {  \
        ptr->serialize(ser);                                \
    }                                                       \
    void deserialize(::arb::serdes::serializer& ser) {      \
        ptr->deserialize(ser);                              \
    }

} // serdes
} // arb
