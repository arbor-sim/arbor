#pragma once

#include <functional>

#include <common_types.hpp>
#include <util/optional.hpp>

namespace nest {
namespace mc {

using sampler_function = std::function<util::optional<time_type>(time_type, double)>;

} // namespace mc
} // namespace nest
