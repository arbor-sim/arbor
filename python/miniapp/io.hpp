#pragma once

#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <common_types.hpp>
#include <util/optional.hpp>
#include <util/path.hpp>

#include "miniapp-base.hpp"

namespace arb {
namespace io {

class usage_error: public std::runtime_error {
public:
    template <typename S>
    usage_error(S&& whatmsg): std::runtime_error(std::forward<S>(whatmsg)) {}
};

/// Helper function for loading a vector of spike times from file
/// Spike times are expected to be in milli seconds floating points
/// On spike-time per line

std::vector<time_type>  get_parsed_spike_times_from_path(arb::util::path path);

std::ostream& operator<<(std::ostream& o, const options& opt);

} // namespace io
} // namespace arb
