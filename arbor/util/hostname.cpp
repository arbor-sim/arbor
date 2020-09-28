#include <optional>
#include <string>

#include "hostname.hpp"

#ifdef __linux__
extern "C" {
    #include <unistd.h>
}
#endif

namespace arb {
namespace util {

#ifdef __linux__
std::optional<std::string> hostname() {
    // Hostnames can be up to 256 characters in length, however on many systems
    // it is limitted to 64.
    char name[256];
    auto result = gethostname(name, sizeof(name));
    if (result) {
        return std::nullopt;
    }
    return std::string(name);
}
#else
std::optional<std::string> hostname() {
    return std::nullopt;
}
#endif

} // namespace util
} // namespace arb

