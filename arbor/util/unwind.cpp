#include <util/unwind.hpp>

#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef WITH_BACKTRACE
#define BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED
#define BOOST_STACKTRACE_USE_ADDR2LINE
#include <boost/stacktrace.hpp>
#endif

namespace arb {
namespace util {

backtrace::backtrace() {
#ifdef WITH_BACKTRACE
    auto bt = boost::stacktrace::basic_stacktrace{};
    for (const auto& f: bt) {
        frames_.push_back(source_location{f.name(), f.source_file(), f.source_line()});
    }
#endif
}

std::ostream& operator<<(std::ostream& out, const backtrace& trace) {
#ifdef WITH_BACKTRACE
    out << "Backtrace:\n";
    int ix = 0;
    for (const auto& f: trace.frames_) {
        out << std::setw(8) << ix << " " << f.func << " (" << f.file << ":" << f.line << ")\n";
        ix++;
    }
#endif
    return out;
}

backtrace& backtrace::pop(std::size_t n) {
    auto end = frames_.begin();
    for(std::size_t ix = 0;
        ix < n && end != frames_.end();
        ++ix, ++end) {}
    frames_.erase(frames_.begin(), end);
    return *this;
}

std::string backtrace::to_string() {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

} // namespace util
} // namespace arb
