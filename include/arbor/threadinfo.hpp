#pragma once

#include <string>

// Query underlying threading implementation for information.
// (Stop-gap until we virtualize threading interface.)

namespace arb {

std::string thread_implementation();

} // namespace arb
