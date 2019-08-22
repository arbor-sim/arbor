#pragma once

/* An option class supporting a subset of C++17 std::optional functionality.
 *
 * Difference from C++17 std::optional:
 *
 * Missing functionality (to be added as required):
 *
 *   1. `constexpr` constructors.
 *
 *   2. Comparison operators other than `operator==`.
 *
 *   3. `std::hash` overload.
 *
 *   4. `swap()` method and ADL-available `swap()` function.
 *
 *   5. In-place construction with `std::in_place_t` tags or equivalent.
 *
 *   5. No `make_optional` function (but see `just` below).
 *
 * Additional/differing functionality:
 *
 *   1. Optional references.
 *
 *      `util::optional<T&>` acts as a value-like wrapper about a possible
 *      reference of type T&. Methods such as `value()` or `value_or()`
 *      return this reference.
 *
 *   2. Optional void.
 *
 *      Included primarily for ease of generic programming with `optional`.
 *
 *   3. `util::just`
 *
 *      This function acts like the value-constructing `std::make_optional<T>(T&&)`,
 *      except that it will return an optional<T&> if given an lvalue T as an argument.
 */

#include <type_traits>
#include <stdexcept>
#include <utility>

#include <arbor/util/uninitialized.hpp>

namespace arb {
namespace util {

template <typename X> struct optional;

struct optional_unset_error: std::runtime_error {
    explicit optional_unset_error(const std::string& what_str)
        : std::runtime_error(what_str)
    {}

    optional_unset_error()
        : std::runtime_error("optional value unset")
    {}
};

struct nullopt_t {};
constexpr nullopt_t nullopt{};

namespace detail {
    template <typename Y>
    struct lift_type {
        using type = optional<Y>;
    };

    template <typename Y>
    struct lift_type<optional<Y>> {
        using type = optional<Y>;
    };

    template <typename Y>
    using lift_type_t = typename lift_type<Y>::type;

    struct optional_tag {};

    template <typename X>
    using is_optional = std::is_base_of<optional_tag, std::decay_t<X>>;

    template <typename D, typename X>
    struct wrapped_type_impl {
        using type = X;
    };

    template <typename D, typename X>
    struct wrapped_type_impl<optional<D>, X> {
        using type = D;
    };

    template <typename X>
    struct wrapped_type {
        using type = typename wrapped_type_impl<std::decay_t<X>, X>::type;
    };

    template <typename X>
    using wrapped_type_t = typename wrapped_type<X>::type;

    template <typename X>
    struct optional_base: detail::optional_tag {
        template <typename Y> friend struct optional;

    protected:
        using data_type = util::uninitialized<X>;
        using rvalue_reference = typename data_type::rvalue_reference;
        using const_rvalue_reference = typename data_type::const_rvalue_reference;

    public:
        using value_type = X;
        using reference = typename data_type::reference;
        using const_reference = typename data_type::const_reference;
        using pointer = typename data_type::pointer;
        using const_pointer = typename data_type::const_pointer;

    protected:
        bool set;
        data_type data;

        optional_base() : set(false) {}

        template <typename T>
        optional_base(bool set_, T&& init) : set(set_) {
            if (set) {
                data.construct(std::forward<T>(init));
            }
        }

        reference       ref()       { return data.ref(); }
        const_reference ref() const { return data.cref(); }

        void assert_set() const {
            if (!set) {
                throw optional_unset_error();
            }
        }

    public:
        ~optional_base() {
            if (set) {
                data.destruct();
            }
        }

        pointer operator->() { return data.ptr(); }
        const_pointer operator->() const { return data.cptr(); }

        reference operator*() { return ref(); }
        const_reference operator*() const { return ref(); }

        explicit operator bool() const { return set; }

        template <typename Y>
        bool operator==(const Y& y) const {
            return set && ref()==y;
        }

        template <typename Y>
        bool operator==(const optional<Y>& o) const {
            return (set && o.set && ref()==o.ref()) || (!set && !o.set);
        }

        void reset() {
            if (set) {
                data.destruct();
            }
            set = false;
        }

        template <typename Y>
        void emplace(Y&& y) {
            if (set) {
                data.destruct();
            }
            data.construct(std::forward<Y>(y));
            set = true;
        }
    };

    // type utilities
    template <typename T>
    using enable_unless_optional_t = std::enable_if_t<!is_optional<T>::value>;

    // avoid nonnull address warnings when using operator| with e.g. char array constants
    template <typename T>
    bool decay_bool(const T* x) { return static_cast<bool>(x); }

    template <typename T>
    bool decay_bool(const T& x) { return static_cast<bool>(x); }

} // namespace detail

template <typename X>
struct optional: detail::optional_base<X> {
    using base = detail::optional_base<X>;
    using base::set;
    using base::ref;
    using base::reset;
    using base::data;
    using base::assert_set;

    optional() noexcept: base() {}
    optional(nullopt_t) noexcept: base() {}

    optional(const X& x)
        noexcept(std::is_nothrow_copy_constructible<X>::value): base(true, x) {}

    optional(X&& x)
        noexcept(std::is_nothrow_move_constructible<X>::value): base(true, std::move(x)) {}

    optional(const optional& ot): base(ot.set, ot.ref()) {}

    template <typename T>
    optional(const optional<T>& ot)
        noexcept(std::is_nothrow_constructible<X, T>::value): base(ot.set, ot.ref()) {}

    optional(optional&& ot)
        noexcept(std::is_nothrow_move_constructible<X>::value): base(ot.set, std::move(ot.ref())) {}

    template <typename T>
    optional(optional<T>&& ot)
        noexcept(std::is_nothrow_constructible<X, T&&>::value): base(ot.set, std::move(ot.ref())) {}

    optional& operator=(nullopt_t) {
        reset();
        return *this;
    }

    template <
        typename Y,
        typename = detail::enable_unless_optional_t<Y>
    >
    optional& operator=(Y&& y) {
        if (set) {
            ref() = std::forward<Y>(y);
        }
        else {
            set = true;
            data.construct(std::forward<Y>(y));
        }
        return *this;
    }

    optional& operator=(const optional& o) {
        if (set) {
            if (o.set) {
                ref() = o.ref();
            }
            else {
                reset();
            }
        }
        else {
            set = o.set;
            if (set) {
                data.construct(o.ref());
            }
        }
        return *this;
    }

    template <
        typename Y = X,
        typename = std::enable_if_t<
            std::is_move_assignable<Y>::value &&
            std::is_move_constructible<Y>::value
        >
    >
    optional& operator=(optional&& o) {
        if (set) {
            if (o.set) {
                ref() = std::move(o.ref());
            }
            else reset();
        }
        else {
            set = o.set;
            if (set) {
                data.construct(std::move(o.ref()));
            }
        }
        return *this;
    }

    X& value() & {
        return assert_set(), ref();
    }

    const X& value() const& {
        return assert_set(), ref();
    }

    X&& value() && {
        return assert_set(), std::move(ref());
    }

    const X&& value() const&& {
        return assert_set(), std::move(ref());
    }

    template <typename T>
    X value_or(T&& alternative) const& {
        return set? value(): static_cast<X>(std::forward<T>(alternative));
    }

    template <typename T>
    X value_or(T&& alternative) && {
        return set? std::move(value()): static_cast<X>(std::forward<T>(alternative));
    }
};

template <typename X>
struct optional<X&>: detail::optional_base<X&> {
    using base=detail::optional_base<X&>;
    using base::set;
    using base::ref;
    using base::data;
    using base::reset;
    using base::assert_set;

    optional() noexcept: base() {}
    optional(nullopt_t) noexcept: base() {}
    optional(X& x) noexcept: base(true, x) {}

    template <typename T>
    optional(optional<T&>& ot) noexcept: base(ot.set, ot.ref()) {}

    optional& operator=(nullopt_t) {
        reset();
        return *this;
    }

    template <typename Y>
    optional& operator=(Y& y) {
        set = true;
        data.construct(y);
        return *this;
    }

    template <typename Y>
    optional& operator=(optional<Y&>& o) {
        set = o.set;
        if (o.set) {
           data.construct(o.value());
        }
        return *this;
    }

    X& value() {
        return assert_set(), ref();
    }

    const X& value() const {
        return assert_set(), ref();
    }

    X& value_or(X& alternative) & {
        return set? ref(): alternative;
    }

    const X& value_or(const X& alternative) const& {
        return set? ref(): alternative;
    }

    template <typename T>
    const X value_or(const T& alternative) && {
        return set? ref(): static_cast<X>(alternative);
    }
};

template <>
struct optional<void>: detail::optional_base<void> {
    using base = detail::optional_base<void>;
    using base::assert_set;
    using base::set;
    using base::reset;

    optional(): base() {}
    optional(nullopt_t): base() {}

    template <typename T>
    optional(T): base(true, true) {}

    template <typename T>
    optional(const optional<T>& o): base(o.set, true) {}

    optional& operator=(nullopt_t) {
        reset();
        return *this;
    }

    template <typename T>
    optional& operator=(T) {
        set = true;
        return *this;
    }

    template <typename T>
    optional& operator=(const optional<T>& o) {
        set = o.set;
        return *this;
    }

    template <typename Y>
    bool operator==(const Y& y) const { return false; }

    bool operator==(const optional<void>& o) const {
        return (set && o.set) || (!set && !o.set);
    }

    void value() const { assert_set(); }

    template <typename T>
    void value_or(T) const {} // nop
};

template <typename X>
optional<X> just(X&& x) {
    return optional<X>(std::forward<X>(x));
}

} // namespace util
} // namespace arb
