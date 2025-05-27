#pragma once

#include <algorithm>
#include <cstring>
#include <iterator>
#include <optional>
#include <utility>
#include <type_traits>

#include "util/meta.hpp"

// Convenience views, algorithms for maps and map-like containers.

namespace arb {
namespace util {

// Is a container/sequence a map?
namespace maputil_impl {
    template <
        typename C,
        typename seq_value = typename sequence_traits<C>::value_type,
        typename K = std::tuple_element_t<0, seq_value>,
        typename find_value = std::decay_t<decltype(*std::declval<C>().find(std::declval<K>()))>
    >
    struct assoc_test: std::integral_constant<bool, std::is_same<seq_value, find_value>::value> {};
}

template <typename Seq, typename = void>
struct is_associative_container: std::false_type {};

template <typename Seq>
struct is_associative_container<Seq, std::void_t<maputil_impl::assoc_test<Seq>>>: maputil_impl::assoc_test<Seq> {};

// Find value in a sequence of key-value pairs or in a key-value assocation map, with
// optional explicit comparator.
//
// If no comparator is given, and the container is associative, use the `find` method, otherwise
// perform a linear search.
//
// Returns optional<value> or optional<value&>. A reference optional is returned if:
//   1. the sequence is an lvalue reference, and
//   2. if the deduced return type from calling `get` on an entry from the sequence is an lvalue reference.
namespace maputil_impl {
    // import std::get and std::begin for ADL below.
    using std::begin;
    using std::get;

    // use linear search
    template <
        typename Seq,
        typename Key,
        typename Eq = std::equal_to<>,
        typename Ret = std::remove_reference_t<decltype(get<1>(*begin(std::declval<Seq&&>())))>
    >
    std::optional<Ret> value_by_key(std::false_type, Seq&& seq, const Key& key, Eq eq=Eq{}) {
        for (auto&& entry: seq) {
            if (eq(get<0>(entry), key)) {
                return get<1>(entry);
            }
        }
        return std::nullopt;
    }

    // use linear search
    template <
        typename Seq,
        typename Key,
        typename Eq = std::equal_to<>,
        typename Ret = std::remove_reference_t<decltype(get<1>(*begin(std::declval<Seq&&>())))>
    >
    Ret value_by_key_or(std::false_type, Seq&& seq, const Key& key, Ret def, Eq eq=Eq{}) {
        for (auto&& entry: seq) {
            if (eq(get<0>(entry), key)) {
                return get<1>(entry);
            }
        }
        return def;
    }

    template <
        typename Seq,
        typename Key,
        typename Eq = std::equal_to<>,
        typename Ret = std::remove_reference_t<decltype(get<1>(*begin(std::declval<Seq&&>())))>
    >
    Ret* ptr_by_key(std::false_type, Seq&& seq, const Key& key, Eq eq=Eq{}) {
        for (auto&& entry: seq) {
            if (eq(get<0>(entry), key)) {
                return &get<1>(entry);
            }
        }
        return nullptr;
    }

    // use map find
    template <
        typename Assoc,
        typename Key,
        typename FindRet = decltype(std::declval<Assoc&&>().find(std::declval<Key>())),
        typename Ret = std::remove_reference_t<decltype(get<1>(*std::declval<FindRet>()))>
    >
    std::optional<Ret> value_by_key(std::true_type, Assoc&& map, const Key& key) {
        auto it = map.find(key);
        if (it!=std::end(map)) {
            return get<1>(*it);
        }
        return std::nullopt;
    }

    template <
        typename Assoc,
        typename Key,
        typename FindRet = decltype(std::declval<Assoc&&>().find(std::declval<Key>())),
        typename Ret = std::remove_reference_t<decltype(get<1>(*std::declval<FindRet>()))>
    >
    Ret value_by_key_or(std::true_type, Assoc&& map, const Key& key, Ret def) {
        auto it = map.find(key);
        if (it!=std::end(map)) {
            return get<1>(*it);
        }
        return def;
    }

    template <
        typename Assoc,
        typename Key,
        typename FindRet = decltype(std::declval<Assoc&&>().find(std::declval<Key>())),
        typename Ret = std::remove_reference_t<decltype(get<1>(*std::declval<FindRet>()))>
    >
    Ret* ptr_by_key(std::true_type, Assoc&& map, const Key& key) {
        auto it = map.find(key);
        return it!=std::end(map)? &get<1>(*it): nullptr;
    }
}

// Return copy of value associated with key, wrapped in std::optional, or std::nullopt.
template <typename C, typename Key, typename Eq>
auto value_by_key(C&& c, const Key& k, Eq eq) {
    return maputil_impl::value_by_key(std::false_type{}, std::forward<C>(c), k, eq);
}

template <typename C, typename Key>
auto value_by_key(C&& c, const Key& k) {
    return maputil_impl::value_by_key(
        std::integral_constant<bool, is_associative_container<C>::value>{},
        std::forward<C>(c), k);
}

// Return copy of value associated with key; if absent return default
template <typename C, typename Key, typename Val, typename Eq>
auto value_by_key_or(C&& c, const Key& k, Val&& v, Eq eq) {
    return maputil_impl::value_by_key_or(std::false_type{}, std::forward<C>(c), k, v, eq);
}

template <typename C,
          typename Key,
          typename Val = std::remove_reference_t<decltype(std::get<1>(*begin(std::declval<C&&>())))>>
auto value_by_key_or(C&& c, const Key& k, Val&& v) {
    return maputil_impl::value_by_key_or(
        std::integral_constant<bool, is_associative_container<C>::value>{},
        std::forward<C>(c), k, v);
}

// Return pointer to value associated with key, or nullptr.
template <typename C, typename Key, typename Eq>
auto ptr_by_key(C&& c, const Key& k, Eq eq) {
    return maputil_impl::ptr_by_key(std::false_type{}, std::forward<C>(c), k, eq);
}

template <typename C, typename Key>
auto ptr_by_key(C&& c, const Key& k) {
    return maputil_impl::ptr_by_key(
        std::integral_constant<bool, is_associative_container<C>::value>{},
        std::forward<C>(c), k);
}

// Find the index into an ordered sequence of a value by binary search;
// returns optional<size_type> for the size_type associated with the sequence.
template <typename C, typename Key>
std::optional<typename sequence_traits<C>::difference_type> binary_search_index(const C& c, const Key& key) {
    auto strict = strict_view(c);
    auto it = std::lower_bound(strict.begin(), strict.end(), key);
    return it!=strict.end() && key==*it? std::optional(std::distance(strict.begin(), it)): std::nullopt;
}

// As binary_search_index above, but compare key against the proj(x) for elements
// x in the sequence. The image of proj applied to the sequence must be monotonically
// increasing.
template <typename C, typename Key, typename Proj>
std::optional<typename sequence_traits<C>::difference_type> binary_search_index(const C& c, const Key& key, const Proj& proj) {
    auto strict = strict_view(c);
    auto it = std::ranges::lower_bound(strict, key, std::ranges::less{}, proj);
    return (it != strict.end() && key == proj(*it)) ? std::optional(std::distance(strict.begin(), it)) : std::nullopt;
}

} // namespace util
} // namespace arb
