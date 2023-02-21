#include <string>
#include <sstream>

namespace arb {
namespace util {
// Substitute instances of '{}' in the format string with the following parameters,
// using default std::ostream formatting.

namespace impl {
    inline void pprintf_(std::ostringstream& o, const char* s) {
        o << s;
    }

    template <typename T, typename... Tail>
    void pprintf_(std::ostringstream& o, const char* s, T&& value, Tail&&... tail) {
        const char* t = s;
        while (*t && !(*t=='{' && t[1]=='}')) {
            ++t;
        }
        o.write(s, t-s);
        if (*t) {
            o << std::forward<T>(value);
            pprintf_(o, t+2, std::forward<Tail>(tail)...);
        }
    }
} // namespace impl

template <typename... Args>
std::string pprintf(const char *s, Args&&... args) {
    std::ostringstream o;
    impl::pprintf_(o, s, std::forward<Args>(args)...);
    return o.str();
}

} // namespace util
} // namespace arb

