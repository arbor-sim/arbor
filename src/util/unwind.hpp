#pragma once

#ifdef WITH_UNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#endif

#include <cxxabi.h>
#include <string>
#include <vector>

namespace nest {
namespace mc {
namespace util {

#ifdef WITH_UNWIND

// helper function that demangles a function name
// if s is not a valid mangled C++ name
//   return s
// else
//   return demangled name
std::string demangle(std::string s);

///  Represents a source code location as a function name and address
struct source_location {
    std::string name;
    unw_word_t position;
};

///  Builds a stack trace when constructed.
///  The trace can then be printed, or accessed via the stack() member function.
class backtrace {
public:
    /// the default constructor will build and storethe strack trace.
    backtrace();

    //
    void print(bool stop_at_main=true) const;
    const std::vector<source_location>& frames() const { return frames_; }

private:
    std::vector<source_location> frames_;
};

#else
//
//  provide empty stack trace proxy in case libunwind is not supported
//
struct source_location {
    std::string name;
    std::size_t position;
};

class backtrace {
public:
    void print(bool stop_at_main=true) const { }
    std::vector<source_location> frames() const {
        return {};
    }
};

#endif

} // namespace util
} // namespace mc
} // namespace nest
