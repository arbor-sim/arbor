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

#ifdef WITH_BACKTRACE
backtrace::backtrace() {
    auto bt = boost::stacktrace::basic_stacktrace{};
    for (const auto& f: bt) {
        frames_.push_back(source_location{f.name(), f.source_file(), f.source_line()});
    }

}

std::ostream& operator<<(std::ostream& out, const backtrace& trace) {
    out << "Backtrace:\n";
    int ix = 0;
    for (const auto& f: trace.frames_) {
        out << std::setw(8) << ix << " " << f.func << " (" << f.file << ":" << f.line << ")\n";
        ix++;
    }
    return out;
}

#else

backtrace::backtrace() = default;

std::ostream& operator<<(std::ostream& out, const backtrace&) {
    out << "Backtrace disabled\n";
    return out;
}

#endif

backtrace& backtrace::pop(std::size_t n) {
    frames_.erase(frames_.begin(),
                  frames_.begin() + std::min(n, frames_.size()));
    return *this;
}

std::string backtrace::to_string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

} // namespace util
} // namespace arb
