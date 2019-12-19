#pragma once

// Maps keyed by type, with a fixed set of keys (static_typed_map) or
// an arbitrary set (dynamic_typed_map).
//
// The first template parameter maps the key type to its corresponding
// value type.

#include <tuple>
#include <typeindex>
#include <unordered_map>

#include <arbor/util/any.hpp>

namespace arb {

template <template <typename> typename E>
struct dynamic_typed_map {
    template <typename T>
    E<T>& get() {
        arb::util::any& store_entry = tmap_[std::type_index(typeid(T))];
        if (!store_entry.has_value()) {
            store_entry = arb::util::any(E<T>{});
        }

        return arb::util::any_cast<E<T>&>(store_entry);
    }

    template <typename T>
    bool has() const { return tmap_.count(std::type_index(typeid(T))); }

    template <typename T>
    const E<T>& get() const {
        return arb::util::any_cast<const E<T>&>(tmap_.at(std::type_index(typeid(T))));
    }

private:
    std::unordered_map<std::type_index, arb::util::any> tmap_;
};

template <template <typename> typename E, typename... Keys>
struct static_typed_map {
    template <typename T>
    E<T>& get() {
        return std::get<index<T, Keys...>()>(tmap_);
    }

    template <typename T>
    const E<T>& get() const {
        return std::get<index<T, Keys...>()>(tmap_);
    }

    template <typename T>
    constexpr bool has() const { return index<T, Keys...>()<sizeof...(Keys); }

private:
    std::tuple<E<Keys>...> tmap_;

    template <typename T>
    static constexpr int index() { return 1; }

    template <typename T, typename H, typename... A>
    static constexpr int index() {
        return std::is_same<H, T>::value? 0: 1+index<T, A...>();
    }
};

} // namespace arb
