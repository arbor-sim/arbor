#include <util/scope_exit.hpp>
#include <util/path.hpp>

// POSIX header
#include <glob.h>

namespace nest {
namespace mc {
namespace util {

std::vector<path> glob(const std::string& pattern) {
    std::vector<path> paths;

    int flags = GLOB_MARK | GLOB_NOCHECK;
#if defined(GLOB_TILDE)
    flags |= GLOB_TILDE;
#endif
#if defined(GLOB_TILDE)
    flags |= GLOB_BRACE;;
#endif

    glob_t matches;
    auto r = ::glob(pattern.c_str(), flags, nullptr, &matches);
    auto glob_guard = on_scope_exit([&]() { ::globfree(&matches); });

    if (r==GLOB_NOSPACE) {
        throw std::bad_alloc{};
    }
    else if (r==0) {
        // success
        paths.reserve(matches.gl_pathc);
        for (auto pathp = matches.gl_pathv; *pathp; ++pathp) {
            paths.push_back(path{*pathp});
        }
    }

    return paths;
}

} // namespace util
} // namespace mc
} // namespace nest

