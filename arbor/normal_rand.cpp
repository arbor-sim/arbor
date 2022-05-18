
#include <cstring>
#include <iostream>

#include <Random123/boxmuller.hpp>
#include <Random123/threefry.h>

#include <arbor/normal_rand.hpp>

namespace arb {
namespace math {

double normal_rand(
    std::uint64_t i_,
    std::uint64_t seed,
    std::uint64_t gid,
    double time,
    std::uint64_t mech_id,
    std::uint64_t mech_inst,
    std::uint64_t var_id) {
    // std::uint64_t user_seed = 0;
    using rng = r123::Threefry4x64_R<12>;
    using counter_type = typename rng::key_type;
    // using counter_type = r123array4x64;
    using key_type = typename rng::key_type;

    static_assert(std::is_same<typename counter_type::value_type, std::uint64_t>::value, "64 bit width");
    static_assert(std::is_same<typename key_type::value_type, std::uint64_t>::value, "64 bit width");
    static_assert(counter_type::static_size == 4, "size of array");
    static_assert(key_type::static_size == 4, "size of array");

    counter_type c{mech_inst, i_, 0, 0};
    key_type k{seed, mech_id, var_id, gid};
    std::memcpy(&k[2], &time, sizeof(double));

    const auto r = rng{}(c, k);
    const auto [a, b] = r123::boxmuller(r[0], r[1]);

    std::cout << "generating random number: "
              << "\n  i_        = " << i_
              << "\n  seed      = " << seed
              << "\n  gid       = " << gid
              << "\n  time      = " << time
              << "\n  mech_id   = " << mech_id
              << "\n  mech_inst = " << mech_inst
              << "\n  var_id    = " << var_id
              << "\n  r         = " << a
              << std::endl;
    return a;
}

} // namespace math
} // namespace arb
