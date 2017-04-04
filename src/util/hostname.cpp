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
    char name[128];
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

