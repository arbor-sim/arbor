#pragma once

#include <iostream>
#include <cstdint>
#include <string>
#include <vector>

#include <arbor/export.hpp>

namespace arb {
namespace util {

/// Represents a source code location as a function name and address
struct source_location {
    std::string func;
    std::string file;
    std::size_t line;
};

/// Builds a stack trace when constructed.
/// NOTE: if WITH_BACKTRACE is not defined, the methods are empty
class ARB_ARBOR_API backtrace {
public:
    /// the default constructor will build and store the strack trace.
    backtrace();

    std::vector<source_location>& frames() { return frames_; }

    friend std::ostream& operator<<(std::ostream&, const backtrace&);

    // remove the top N=1 frames
    backtrace& pop(std::size_t n=1);

    std::string to_string() const;

private:
    std::vector<source_location> frames_;
};

} // namespace util
} // namespace arb
