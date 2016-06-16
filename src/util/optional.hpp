#pragma once

/* An option class with a monadic interface.
 *
 * The std::option<T> class was proposed for inclusion into C++14, but was
 * ultimately rejected. (See N3672 proposal for details.) This class offers
 * similar functionality, namely a class that can represent a value (or
 * reference), or nothing at all.
 *
 * In addition, this class offers monadic and monoidal bindings, allowing
 * the chaining of operations any one of which might represent failure with
 * an unset optional value.
 *
 * One point of difference between the proposal N3672 and this implementation
 * is the lack of constexpr versions of the methods and constructors.
 */

#include <type_traits>
#include <stdexcept>
#include <utility>

#include "util/meta.hpp"
#include "util/uninitialized.hpp"

namespace nest {
namespace mc {
namespace util {

template <typename X> struct optional;

struct optional_unset_error: std::runtime_error {
    explicit optional_unset_error(const std::string &what_str)
        : std::runtime_error(what_str)
    {}

    optional_unset_error()
        : std::runtime_error("optional value unset")
    {}
};

struct optional_invalid_dereference: std::runtime_error {
    explicit optional_invalid_dereference(const std::string &what_str)
        : std::runtime_error(what_str)
    {}

    optional_invalid_dereference()
        : std::runtime_error("derefernce of optional<void> value")
    {}
};

struct nothing_t {};
constexpr nothing_t nothing{};

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
    using is_optional = std::is_base_of<optional_tag, typename std::decay<X>::type>;

    template <typename D,typename X>
    struct wrapped_type_impl {
        using type = X;
    };

    template <typename D,typename X>
    struct wrapped_type_impl<optional<D>,X> {
        using type = D;
    };

    template <typename X>
    struct wrapped_type {
       using type = typename wrapped_type_impl<typename std::decay<X>::type,X>::type;
    };

    template <typename X>
    using wrapped_type_t = typename wrapped_type<X>::type;

    template <typename X>
    struct optional_base: detail::optional_tag {
        template <typename Y> friend struct optional;

    protected:
        using data_type = util::uninitialized<X>;

    public:
        using reference_type = typename data_type::reference_type;
        using const_reference_type = typename data_type::const_reference_type;
        using pointer_type = typename data_type::pointer_type;
        using const_pointer_type = typename data_type::const_pointer_type;

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

        reference_type       ref()       { return data.ref(); }
        const_reference_type ref() const { return data.cref(); }

    public:
        ~optional_base() {
            if (set) {
                data.destruct();
            }
        }

        const_pointer_type operator->() const { return data.ptr(); }
        pointer_type       operator->()       { return data.ptr(); }

        const_reference_type operator*() const { return ref(); }
        reference_type       operator*()       { return ref(); }

        reference_type get() {
            // I find this super verbose :(
            if (set) {
                return ref();
            }
            else {
                throw optional_unset_error();
            }
        }

        const_reference_type get() const {
            if (set) {
                return ref();
            }
            else {
                throw optional_unset_error();
            }
        }

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

        template <typename F>
        auto bind(F&& f) -> lift_type_t<decltype(data.apply(std::forward<F>(f)))> {
            using F_result_type = decltype(data.apply(std::forward<F>(f)));
            using result_type = lift_type_t<F_result_type>;

            if (!set) {
                return result_type();
            }

            return bind_impl<result_type, std::is_void<F_result_type>::value>::
                       bind(data, std::forward<F>(f));
        }

        template <typename F>
        auto bind(F&& f) const -> lift_type_t<decltype(data.apply(std::forward<F>(f)))> {
            using F_result_type = decltype(data.apply(std::forward<F>(f)));
            using result_type = lift_type_t<F_result_type>;

            if (!set) {
                return result_type();
            }

            return bind_impl<result_type, std::is_void<F_result_type>::value>::
                       bind(data, std::forward<F>(f));
        }

        template <typename F>
        auto operator>>(F&& f) -> decltype(this->bind(std::forward<F>(f))) {
            return bind(std::forward<F>(f));
        }

        template <typename F>
        auto operator>>(F&& f) const -> decltype(this->bind(std::forward<F>(f))) {
            return bind(std::forward<F>(f));
        }

    private:
        template <typename R, bool F_void_return>
        struct bind_impl {
            template <typename DT,typename F>
            static R bind(DT& d,F&& f) {
                return R(d.apply(std::forward<F>(f)));
            }
        };

        template <typename R>
        struct bind_impl<R,true> {
            template <typename DT,typename F>
            static R bind(DT& d,F&& f) {
                d.apply(std::forward<F>(f));
                return R(true);
            }
        };
    };

    // type utilities
    template <typename T>
    using enable_unless_optional_t = enable_if_t<!is_optional<T>::value>;

} // namespace detail

template <typename X>
struct optional: detail::optional_base<X> {
    using base = detail::optional_base<X>;
    using base::set;
    using base::ref;
    using base::reset;
    using base::data;

    optional(): base() {}
    optional(nothing_t): base() {}

    template <
        typename Y = X,
        typename = enable_if_copy_constructible_t<Y>
    >
    optional(const X& x): base(true, x) {}

    template <
        typename Y = X,
        typename = enable_if_move_constructible_t<Y>
    >
    optional(X&& x): base(true, std::move(x)) {}

    optional(const optional& ot): base(ot.set, ot.ref()) {}

    template <typename T>
    optional(const optional<T>& ot): base(ot.set, ot.ref()) {}

    optional(optional&& ot): base(ot.set, std::move(ot.ref())) {}

    template <typename T>
    optional(optional<T>&& ot): base(ot.set, std::move(ot.ref())) {}

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
        typename = typename std::enable_if<
            std::is_move_assignable<Y>::value &&
            std::is_move_constructible<Y>::value
        >::type
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
};

template <typename X>
struct optional<X&>: detail::optional_base<X&> {
    using base=detail::optional_base<X&>;
    using base::set;
    using base::ref;
    using base::data;

    optional(): base() {}
    optional(nothing_t): base() {}
    optional(X&x): base(true,x) {}

    template <typename T>
    optional(optional<T&>& ot): base(ot.set,ot.ref()) {}

    template <typename Y,typename = typename std::enable_if<!detail::is_optional<Y>()>::type>
    optional& operator=(Y& y) {
        set = true;
        ref() = y;
        return *this;
    }

    template <typename Y>
    optional& operator=(optional<Y&>& o) {
        set = o.set;
        data.construct(o);
        return *this;
    }
};


/* special case for optional<void>, used as e.g. the result of
 * binding to a void function */

template <>
struct optional<void>: detail::optional_base<void> {
    using base = detail::optional_base<void>;
    using base::set;

    optional(): base() {}

    template <typename T>
    optional(T): base(true,true) {}

    template <typename T>
    optional(const optional<T>& o): base(o.set,true) {}
    
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
};


template <typename A,typename B>
typename std::enable_if<
    detail::is_optional<A>::value || detail::is_optional<B>::value,
    optional<
        typename std::common_type<
            detail::wrapped_type_t<A>,
            detail::wrapped_type_t<B>
        >::type
    >
>::type
operator|(A&& a,B&& b) {
    return a ? a : b;
}

template <typename A,typename B>
typename std::enable_if<
    detail::is_optional<A>::value || detail::is_optional<B>::value,
    optional<detail::wrapped_type_t<B>>
>::type
operator&(A&& a,B&& b) {
    using result_type = optional<detail::wrapped_type_t<B>>;
    return a ? b: result_type();
}

inline optional<void> provided(bool condition) {
    return condition ? optional<void>(true) : optional<void>();
}

template <typename X>
optional<X> just(X&& x) {
    return optional<X>(std::forward<X>(x));
}

} // namespace util
} // namespace mc
} // namespace nest
