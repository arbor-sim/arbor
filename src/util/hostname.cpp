#include <string>

#include "hostname.hpp"

#ifdef __linux__
extern "C" {
    #include <unistd.h>
}
#endif

namespace nest {
namespace mc {
namespace util {

#ifdef __linux__
std::string hostname() {
    // Hostnames can be up to 256 characters in length, however on many systems
    // it is limitted to 64.
    char name[256];
    auto result = gethostname(name, sizeof(name));
    return result? "unknown": name;
}
#else
std::string hostname() {
    return "unknown";
}
#endif

} // namespace util
} // namespace mc
} // namespace nest

