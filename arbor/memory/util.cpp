#include <iostream>
#include <string>

#include "util/unwind.hpp"
#include "util.hpp"

namespace arb {
namespace memory {
namespace util {

void log_error(const char* file, int line, const std::string& msg) {
    std::cerr
        << arb::util::backtrace()
        << red("runtime error") << " @ "
        << white(file) << ":" << line << "\n    " << msg << std::endl;
}

} // namespace util
} // namespace memory
} // namespace arb
