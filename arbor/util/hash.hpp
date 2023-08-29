#pragma once

#include <cstdint>
#include <string_view>

namespace arb {
using hash_type = uint64_t;

// Non-cryptographic hash function for mapping strings to internal
// identifiers. Concretely, FNV-1a hash function taken from
//
//   http://www.isthe.com/chongo/tech/comp/fnv/index.html
//
// NOTE: It may be worth it considering different hash functions in
//       the future that have better characteristic, xxHash or Murmur
//       look interesting but are more complex and likely require adding
//       external dependencies.
//       NOTE: this is the obligatory comment on a better hash function
//             that will be here until the end of time.

constexpr hash_type offset_basis = 0xcbf29ce484222325;
constexpr hash_type prime        = 0x100000001b3;

constexpr hash_type internal_hash(std::string_view data) {
    hash_type hash = offset_basis;

    for (uint8_t byte: data) {
        hash = hash ^ byte;
        hash = hash * prime;
    }

    return hash;
}

}
