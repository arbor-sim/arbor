#pragma once

#include <functional>
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
    typename = std::enable_if_t<std::is_nothrow_move_constructible<F>::value>
>
class scope_exit {
    F on_exit;
    bool trigger = true;

public:
    template <
        typename F2,
        typename = std::enable_if_t<std::is_nothrow_constructible<F, F2>::value>
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

// std::function is not nothrow move constructable before C++20, so, er, cheat.
namespace impl {
    template <typename R>
    struct wrap_std_function {
        std::function<R ()> f;

        wrap_std_function() noexcept = default;
        wrap_std_function(const std::function<R ()>& f): f(f) {}
        wrap_std_function(std::function<R ()>&& f): f(std::move(f)) {}
        wrap_std_function(wrap_std_function&& other) noexcept {
            try {
                f = std::move(other.f);
            }
            catch (...) {}
        }

        void operator()() const { f(); }
    };
}

template <typename F>
auto on_scope_exit(F&& f) {
    return scope_exit<std::decay_t<F>>(std::forward<F>(f));
}

template <typename R>
auto on_scope_exit(std::function<R ()> f) {
    return on_scope_exit(impl::wrap_std_function<R>(std::move(f)));
}

} // namespace util
} // namespace arb
