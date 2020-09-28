#pragma once

// Maps keyed by type, with a fixed set of keys (static_typed_map) or
// an arbitrary set (dynamic_typed_map).
//
// The first template parameter maps the key type to its corresponding
// value type.
//
// Example: associating a vector of ints with 'int' and a vector of
// doubles with 'double':
//
//     static_typed_map<std::vector, int, double> m;
//     m.get<int>() = {1, 2, 3};
//     m.get<double>() = {1.2, 2.3};

#include <any>
#include <tuple>
#include <typeindex>
#include <unordered_map>

namespace arb {

template <template <class> class E>
struct dynamic_typed_map {
    // Retrieve value by reference associated with type T; create entry with
    // default value if no entry in map for T.
    template <typename T>
    E<T>& get() {
        std::any& store_entry = tmap_[std::type_index(typeid(T))];
        if (!store_entry.has_value()) {
            store_entry = std::any(E<T>{});
        }

        return std::any_cast<E<T>&>(store_entry);
    }

    // Retrieve value by const reference associated with type T;
    // throw if no entry in map for T.
    template <typename T>
    const E<T>& get() const {
        return std::any_cast<const E<T>&>(tmap_.at(std::type_index(typeid(T))));
    }

    // True if map has an entry for type T.
    template <typename T>
    bool has() const { return tmap_.count(std::type_index(typeid(T))); }

private:
    std::unordered_map<std::type_index, std::any> tmap_;
};

template <template <class> class E, typename... Keys>
struct static_typed_map {
    // Retrieve value by reference associated with type T.
    template <typename T>
    E<T>& get() {
        return std::get<index<T, Keys...>()>(tmap_);
    }

    // Retrieve value by const reference associated with type T.
    template <typename T>
    const E<T>& get() const {
        return std::get<index<T, Keys...>()>(tmap_);
    }

    // True if map has an entry for type T.
    template <typename T>
    constexpr bool has() const { return index<T, Keys...>()<sizeof...(Keys); }

private:
    std::tuple<E<Keys>...> tmap_;

    template <typename T>
    static constexpr std::size_t index() { return 1; }

    template <typename T, typename H, typename... A>
    static constexpr std::size_t index() {
        return std::is_same<H, T>::value? 0: 1+index<T, A...>();
    }
};

} // namespace arb
