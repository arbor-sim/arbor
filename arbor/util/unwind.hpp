#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace arb {
namespace util {

/// Represents a source code location as a function name and address
struct source_location {
    std::string name;
    std::uintptr_t position; // assume that unw_word_t is a unit64_t
};

/// Builds a stack trace when constructed.
/// The trace can then be printed, or accessed via the stack() member function.
/// NOTE: if WITH_UNWIND is not defined, the methods are empty
class backtrace {
public:
    /// the default constructor will build and store the strack trace.
    backtrace();

    /// Creates a new file named backtrace_# where # is a number chosen
    /// The back trace is printed to the file, and a message printed to
    /// std::cerr with the backtrace file name and instructions for how
    /// to post-process it.
    void print(bool stop_at_main=true) const;
    const std::vector<source_location>& frames() const { return frames_; }

    friend std::ostream& operator<<(std::ostream&, const backtrace&);

private:
    std::vector<source_location> frames_;
};

} // namespace util
} // namespace arb
