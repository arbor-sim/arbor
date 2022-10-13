#pragma once

// This file is intended to be included in the source files directly in order
// to be compiled by either the host or the device compiler 

#include <type_traits>

#include <Random123/boxmuller.hpp>
#include <Random123/threefry.h>

#include "backends/rand_fwd.hpp"

namespace arb {
namespace cbprng {

// checks that chosen random number generator has the correct layout in terms
// of number of counter and key fields
struct traits_ {
    using generator_type = r123::Threefry4x64_R<12>;
    using counter_type = typename generator_type::ctr_type;
    using key_type = counter_type;
    using array_type = counter_type;

    static_assert(counter_type::static_size == cache_size());
    static_assert(std::is_same<typename counter_type::value_type, value_type>::value);
    static_assert(std::is_same<typename generator_type::key_type, key_type>::value);
};

// export the main types for counter based random number generation
using generator  = traits_::generator_type;
using array_type = traits_::array_type;

} // namespace cbprng
} // namespace arb

