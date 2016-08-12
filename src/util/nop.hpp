#pragma once

/*
 * Provide object that implicitly converts to
 * a std::function object that does nothing but return a
 * default-constructed type or void.
 */

#include <functional>

namespace nest {
namespace mc {
namespace util {

static struct nop_function_t {
    template <typename R, typename... Args>
    operator std::function<R (Args...)>() const {
        return [](Args...) { return R{}; };
    }

    template <typename... Args>
    operator std::function<void (Args...)>() const {
        return [](Args...) { };
    }
} nop_function;

} // namespace util
} // namespace mc
} // namespace nest
