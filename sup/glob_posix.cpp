// POSIX headers
extern "C" {
#define _POSIX_C_SOURCE 200809L
#include <glob.h>
}

// GLOB_TILDE and GLOB_BRACE are non-standard but convenient and common
// flags for glob().

#ifndef GLOB_TILDE
#define GLOB_TILDE 0
#endif
#ifndef GLOB_BRACE
#define GLOB_BRACE 0
#endif

#include <cerrno>

#include <arbor/util/scope_exit.hpp>
#include <sup/path.hpp>

using arb::util::on_scope_exit;

namespace sup {

std::vector<path> glob(const std::string& pattern) {
    std::vector<path> paths;
    glob_t matches;

    int flags = GLOB_MARK | GLOB_NOCHECK | GLOB_TILDE | GLOB_BRACE;
    auto r = ::glob(pattern.c_str(), flags, nullptr, &matches);
    auto glob_guard = on_scope_exit([&]() { ::globfree(&matches); });

    if (r==GLOB_NOSPACE) {
        throw std::bad_alloc{};
    }
    else if (r==0) {
        // success
        paths.reserve(matches.gl_pathc);
        for (auto pathp = matches.gl_pathv; *pathp; ++pathp) {
            paths.push_back(*pathp);
        }
    }

    return paths;
}

} // namespace sup

