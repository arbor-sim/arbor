#pragma once
#include <cstring>
#include <array>
#include <Random123/boxmuller.hpp>
#include <Random123/threefry.h>

namespace arb {
namespace multicore {

namespace {
constexpr std::size_t cbprng_batch_size = 4;
}

std::array<arb_value_type,cbprng_batch_size> generate_normal_random_values(
    std::uint64_t seed,
    std::uint64_t mech_id,
    std::uint64_t var_id,
    std::uint64_t gid,
    std::uint64_t idx,
    std::uint64_t counter) {

    using rng = r123::Threefry4x64_R<12>;
    using counter_type = typename rng::key_type;
    using key_type = typename rng::key_type;

    static_assert(std::is_same<typename counter_type::value_type, std::uint64_t>::value,
        "64 bit width");
    static_assert(std::is_same<typename key_type::value_type, std::uint64_t>::value,
        "64 bit width");
    static_assert(counter_type::static_size == cbprng_batch_size, "size of array");
    static_assert(key_type::static_size == cbprng_batch_size, "size of array");

    counter_type c{seed, mech_id, var_id, counter};
    key_type k{gid, idx, 0, 0};

    const auto r = rng{}(c, k);
    const auto n0 = r123::boxmuller(r[0], r[1]);
    const auto n1 = r123::boxmuller(r[2], r[3]);
    //std::cout << "generating random number: "
    //          << "\n  seed      = " << seed
    //          << "\n  counter   = " << counter
    //          << "\n  mech_id   = " << mech_id
    //          << "\n  var_id    = " << var_id
    //          << "\n  gid       = " << gid
    //          << "\n  idx       = " << idx
    //          << std::endl;
    return {n0.x, n0.y, n1.x, n1.y};
}

} // namespace multicore
} // namespace arb

