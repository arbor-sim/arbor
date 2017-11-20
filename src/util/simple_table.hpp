#pragma once

#include <iterator>
#include <cstring>
#include <utility>

// Linear lookup of values in a table indexed by a key.
//
// Tables are any sequence of pairs or tuples, with the key as the first element.

namespace arb {
namespace util {

namespace impl {
    struct key_equal {
        template <typename U, typename V>
        bool operator()(U&& u, V&& v) const {
            return std::forward<U>(u)==std::forward<V>(v);
        }

        // special case for C strings:
        bool operator()(const char* u, const char* v) const {
            return !std::strcmp(u, v);
        }
    };
};

// Return pointer to value (second element in entry) in table if key found,
// otherwise nullptr.

template <typename PairSeq, typename Key, typename Eq = impl::key_equal>
auto table_lookup(PairSeq&& seq, const Key& key, Eq eq = Eq{})
    -> decltype(&std::get<1>(*std::begin(seq)))
{
    for (auto&& entry: seq) {
        if (eq(std::get<0>(entry), key)) {
            return &std::get<1>(entry);
        }
    }
    return nullptr;
}

} // namespace util
} // namespace arb
