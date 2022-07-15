#pragma once
#include <Random123/threefry.h>

namespace arb {

using cbprng_generator = r123::Threefry4x64_R<12>;
using cbprng_value_type = cbprng_generator::ctr_type::value_type;
constexpr std::size_t cbprng_batch_size = cbprng_generator::ctr_type::static_size;

} // namespace arb
