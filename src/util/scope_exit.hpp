#pragma once

#include <type_traits>
#include <utility>

// Convenience class for RAII control of resources.

namespace arb {
namespace util {

// `scope_exit` guard object will call provided functional object
// on destruction. The provided functional object must be nothrow
// move constructible.

template <
    typename F,
    typename = typename std::enable_if<std::is_nothrow_move_constructible<F>::value>::type
>
class scope_exit {
    F on_exit;
    bool trigger = true;

public:
    template <
        typename F2,
        typename = typename std::enable_if<std::is_nothrow_constructible<F, F2>::value>::type
    >
    explicit scope_exit(F2&& f) noexcept:
        on_exit(std::forward<F2>(f)) {}

    scope_exit(scope_exit&& other) noexcept:
        on_exit(std::move(other.on_exit))
    {
        other.release();
    }

    void release() noexcept {
        trigger = false;
     }

    ~scope_exit() noexcept(noexcept(on_exit())) {
        if (trigger) on_exit();
     }
};

template <typename F>
scope_exit<typename std::decay<F>::type> on_scope_exit(F&& f) {
    return scope_exit<typename std::decay<F>::type>(std::forward<F>(f));
}

} // namespace util
} // namespace arb
