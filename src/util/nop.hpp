#pragma once

/*
 * Provide object that implicitly converts to
 * a std::function object that does nothing but return a
 * default-constructed type or void.
 */

#include <functional>

namespace arb {
namespace util {

struct nop_function_t {
    template <typename R, typename... Args>
    operator std::function<R (Args...)>() const {
        return [](Args...) { return R{}; };
    }

    template <typename... Args>
    operator std::function<void (Args...)>() const {
        return [](Args...) { };
    }

    // keep clang happy: see CWG issue #253,
    // http://open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#253
    constexpr nop_function_t() {}
};

static constexpr nop_function_t nop_function;

} // namespace util
} // namespace arb
