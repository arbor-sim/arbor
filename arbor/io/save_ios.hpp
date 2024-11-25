#pragma once

// RAII save-and-restore ios formatting flags.

#include <ios>

namespace arb {
namespace io {

struct save_ios_flags {
    std::ios_base& ios;
    std::ios_base::fmtflags flags;

    save_ios_flags(std::ios_base& ios):
       ios(ios), flags(ios.flags()) {}

    save_ios_flags(const save_ios_flags&) = delete;

    ~save_ios_flags() { ios.flags(flags); }
};


} // namespace io
} // namespace arb
