#pragma once

#include <optional>
#include <string_view>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <array>

#include <iostream>

#include <nlohmann/json.hpp>

namespace arb {
namespace serdes {

using key_type = std::string_view;

template <typename T>
struct value_type {
    using type = T;
};

template <>
struct value_type<char*> {
    using type = std::string;
};

template <>
struct value_type<std::string_view> {
    using type = std::string;
};

template <>
struct value_type<float> {
    using type = double;
};

template <>
struct value_type<int> {
    using type = long long;
};

template <>
struct value_type<unsigned> {
    using type = long long;
};

template <>
struct value_type<unsigned long> {
    using type = long long;
};

struct serializer {
    template <typename I>
    serializer(I& i): wrapped{std::make_unique<wrapper<I>>(i)} {}

    template <typename K, typename V>
    void write(const K& k, const std::unordered_map<std::string, V>& qvs) {
        wrapped->begin_write_map(k);
        for (const auto& [q, v]: qvs) write(q, v);
        wrapped->end_write_map();
    }

    template <typename K, typename V>
    void write(const K& k, const std::map<std::string, V>& qvs) {
        wrapped->begin_write_map(k);
        for (const auto& [q, v]: qvs) write(q, v);
        wrapped->end_write_map();
    }

    template <typename K, typename V>
    void write(const K& k, const std::vector<V>& vs) {
        wrapped->begin_write_array(k);
        for (const auto& v: vs) push(v);
        wrapped->end_write_array();
    }

    template <typename K, typename V, std::size_t N>
    void write(const K& k, const std::array<V, N>& vs) {
        wrapped->begin_write_array(k);
        for (const auto& v: vs) push(v);
        wrapped->end_write_array();
    }

    template <typename K, typename V>
    void write(const K& k, const V& v) {
        using T = typename std::decay<V>::type;
        if constexpr (std::is_same<std::string, T>::value) {
            wrapped->write(k, std::string{v});
        }
        else if constexpr (std::is_same<char*, T>::value) {
            wrapped->write(k, std::string{v});
        }
        else if constexpr (std::is_same<std::string_view, T>::value) {
            wrapped->write(k, std::string{v});
        }
        else if constexpr (std::is_floating_point<T>::value) {
            wrapped->write(k, double{v});
        }
        else if constexpr (std::is_integral<T>::value) {
            wrapped->write(k, static_cast<long long>(v));
        }
        else {
            wrapped->begin_write_map(k);
            v.serialize(*this);
            wrapped->end_write_map();
        }
    }

    template <typename V>
    void push(const V& v) {
        using T = typename std::decay<V>::type;

        if constexpr (std::is_same<std::string, T>::value) {
            wrapped->push(std::string{v});
        }
        else if constexpr (std::is_same<char*, T>::value) {
            wrapped->push(std::string{v});
        }
        else if constexpr (std::is_same<std::string_view, T>::value) {
            wrapped->push(std::string{v});
        }
        else if constexpr (std::is_floating_point<T>::value) {
            wrapped->push(double{v});
        }
        else if constexpr (std::is_integral<T>::value) {
            wrapped->push(static_cast<long long>(v));
        }
    }

    template <typename K, typename V>
    void read(const K& k, V& v) {
        using T = typename std::decay<V>::type;
        if constexpr (std::is_same<std::string, T>::value) {
            wrapped->read(k, v);
        }
        else if constexpr (std::is_floating_point<T>::value) {
            double tmp;
            wrapped->read(k, tmp);
            v = tmp;
        }
        else if constexpr (std::is_integral<T>::value) {
            long long tmp;
            wrapped->read(k, tmp);
            v = tmp;
        }
        else {
            wrapped->begin_read_map(k);
            v.deserialize(*this);
            wrapped->end_read_map();
        }
    }

    template <typename K, typename V>
    void read(const K& k, std::unordered_map<std::string, V>& kvs) {
        V val;
        wrapped->begin_read_map(k);
        for(;;) {
            auto key = wrapped->next_key();
            if (!key) break;
            read(*key, val);
            kvs[*key] = val;
        }
        wrapped->end_read_map();
    }

    template <typename K, typename V>
    void read(const K& k, std::map<std::string, V>& kvs) {
        V val;
        wrapped->begin_read_map(k);
        for(;;) {
            auto key = wrapped->next_key();
            if (!key) break;
            read(*key, val);
            kvs[*key] = val;
        }
        wrapped->end_read_map();
    }

    template <typename K, typename V>
    void read(const K& k, std::vector<V>& vs) {
        wrapped->begin_read_array(k);
        V val;
        for(;;) {
            auto key = wrapped->next_key();
            if (!key) break;
            read(*key, val);
            vs.push_back(val);
        }
        wrapped->end_read_array();
    }

    template <typename K, typename V, std::size_t N>
    void read(const K& k, std::array<V, N>& vs) {
        wrapped->begin_read_array(k);
        V val;
        for(int ix = 0; ix < N; ++ix) {
            auto key = wrapped->next_key();
            if (!key) break;
            read(*key, val);
            vs[ix] = val;
        }
        wrapped->end_read_array();
    }

private:
    struct interface {
        virtual void write(key_type, std::string) = 0;
        virtual void write(key_type, double) = 0;
        virtual void write(key_type, long long) = 0;

        virtual void read(key_type, std::string&) = 0;
        virtual void read(key_type, double&) = 0;
        virtual void read(key_type, long long&) = 0;

        virtual void push(std::string) = 0;
        virtual void push(double) = 0;
        virtual void push(long long) = 0;

        virtual std::optional<std::string> next_key() = 0;

        virtual void begin_write_map(key_type) = 0;
        virtual void end_write_map() = 0;
        virtual void begin_write_array(key_type) = 0;
        virtual void end_write_array() = 0;

        virtual void begin_read_map(key_type) = 0;
        virtual void end_read_map() = 0;
        virtual void begin_read_array(key_type) = 0;
        virtual void end_read_array() = 0;

        virtual ~interface() = default;
    };

    template <typename I>
    struct wrapper: interface {
        wrapper(I& i): inner(i) {}
        I& inner;

        void write(key_type k, std::string v) override { inner.write(k, v); }
        void write(key_type k, double v) override { inner.write(k, v); }
        void write(key_type k, long long v) override { inner.write(k, v); }

        void read(key_type k, std::string& v) override { inner.read(k, v); };
        void read(key_type k, long long& v) override { inner.read(k, v); };
        void read(key_type k, double& v) override { inner.read(k, v); };

        void push(std::string v) override { inner.push(v); }
        void push(double v) override { inner.push(v); }
        void push(long long v) override { inner.push(v); }

        std::optional<std::string> next_key() override { return inner.next_key(); }

        void begin_write_map(key_type k) override { inner.begin_write_map(k); }
        void end_write_map() override { inner.end_write_map(); }

        void begin_write_array(key_type k) override { inner.begin_write_array(k); }
        void end_write_array() override { inner.end_write_array(); }

        void begin_read_map(key_type k) override { inner.begin_read_map(k); }
        void end_read_map() override { inner.end_read_map(); }

        void begin_read_array(key_type k) override { inner.begin_read_array(k); }
        void end_read_array() override { inner.end_read_array(); }

        virtual ~wrapper() = default;
    };

    std::unique_ptr<interface> wrapped;
};

struct json_serdes {
    nlohmann::json data;
    nlohmann::json::json_pointer ptr{""};
    std::optional<decltype(data.items().begin())> iter;
    std::optional<decltype(data.items().end())> stop;

    template <typename V>
    void write(key_type k, const V& v) { data[ptr / std::string(k)] = v; }

    template <typename V>
    void read(key_type k, V& v) { data[ptr / std::string(k)].get_to(v); }

    template <typename V>
    void push(const V& v) { data[ptr].push_back(v); }

    std::optional<std::string> next_key() {
        if (iter && stop && iter != stop) {
            auto key = iter.value().key();
            iter.value()++;
            return {key};
        }
        return {};
    }

    void begin_write_map(key_type k) { ptr /= std::string(k); }
    void end_write_map() { ptr.pop_back(); }
    void begin_write_array(key_type k) {  ptr /= std::string(k); }
    void end_write_array() { ptr.pop_back(); }

    void begin_read_map(key_type k) { ptr /= std::string(k); iter = data[ptr].items().begin(); stop = data[ptr].items().end(); }
    void end_read_map() { ptr.pop_back(); iter = {}; }
    void begin_read_array(key_type k) {  ptr /= std::string(k); iter = data[ptr].items().begin(); stop = data[ptr].items().end(); }
    void end_read_array() { ptr.pop_back(); iter = {}; }
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

#define ARB_SERDES_ENABLE( ...) \
    void serialize(::arb::serdes::serializer& ser) const { \
        ARB_SERDES_EXPAND(ARB_SERDES_PUT(ARB_SERDES_WRITE, __VA_ARGS__)) \
    } \
    void deserialize(::arb::serdes::serializer& ser) { \
        ARB_SERDES_EXPAND(ARB_SERDES_PUT(ARB_SERDES_READ, __VA_ARGS__)) \
    }

} // serdes
} // arb
