#pragma once

/*! \file optional.h
 *  \brief An option class with a monadic interface.
 *
 *  The std::option<T> class was proposed for inclusion into C++14, but was
 *  ultimately rejected. (See N3672 proposal for details.) This class offers
 *  similar functionality, namely a class that can represent a value (or
 *  reference), or nothing at all.
 *
 *  In addition, this class offers monadic and monoidal bindings, allowing
 *  the chaining of operations any one of which might represent failure with
 *  an unset optional value.
 *
 *  One point of difference between the proposal N3672 and this implementation
 *  is the lack of constexpr versions of the methods and constructors.
 */

#include <type_traits>
#include <stdexcept>
#include <utility>

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

    optional_invalid_dereference()\
        : std::runtime_error("derefernce of optional<void> value")
    {}
};

struct nothing_t {};
constexpr nothing_t nothing{};

namespace detail {
    // the only change I make is to add an underscore after lift_type_
    template <typename Y>
    struct lift_type_
    { using type=optional<Y>; };

    template <typename Y>
    struct lift_type_<optional<Y>>
    { using type=optional<Y>; };

    // ... then use an alias template to remove the need for a ::type in user code
    // this is in the same vain as the metafunctions like std::decay_t that were introduced in C++14
    template <typename Y>
    using lift_type = typename lift_type_<Y>::type;

    struct optional_tag {};

    // constexpr function instead
    template <typename X>
    constexpr bool is_optional() {
        return std::is_base_of<optional_tag, typename std::decay<X>::type>::value;
    }

    // you were kind of using the same pattern here...
    template <typename D,typename X>
    struct wrapped_type_
    { using type=X; };

    template <typename D,typename X>
    struct wrapped_type_<optional<D>,X>
    { using type=D; };

    // but we can simplify the following with an alias template
    //template <typename X> struct wrapped_type { using type=typename wrapped_type_<typename std::decay<X>::type,X>::type; };
    template <typename X>
    using wrapped_type =
        typename wrapped_type_<typename std::decay<X>::type, X>::type;

    template <typename X>
    struct optional_base: detail::optional_tag {
        template <typename Y> friend struct optional;

    protected:
        // D is used throughout, so maybe give it a more descriptive name like unitinitialized_type ?
        // whatever, I have the style of using CamelCase for template parameters, and lower case with underscore
        // for "typedefed" types inside the class
        using D=util::uninitialized<X>;

    public:
        // sorry, I like spaces around = signs and spaces after commas
        using reference_type = typename D::reference_type;
        using const_reference_type = typename D::const_reference_type;
        using pointer_type = typename D::pointer_type;
        using const_pointer_type = typename D::const_pointer_type;

    protected:
        bool set;
        D data;

        // don't be afraid of vertical space
        optional_base() : set(false)
        {}

        // I am not too fussy, you could get away with this...
        template <typename T>
        optional_base(bool set_, T&& init) : set(set_) {
            if (set) {
                data.construct(std::forward<T>(init));
            }
        }

        // I could be persuaded to go one line for these, but a bit of vertical alignment helps
        reference_type       ref()       { return data.ref(); }
        const_reference_type ref() const { return data.cref(); }

    public:
        // I know that this is annoying..
        //~optional_base() { if (set) data.destruct(); }
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

        // how about the following as a compromise for simple one-liners?
        explicit operator bool() const
        { return set; }

        // though this is just one more line...
        template <typename Y>
        bool operator==(const Y &y) const {
            return set && ref()==y;
        }

        template <typename Y>
        bool operator==(const optional<Y> &o) const {
            return (set && o.set && ref()==o.ref()) || (!set && !o.set);
        }

        void reset() {
            if (set) {
                data.destruct();
            }
            set=false;
        }

        // see how we remove another typename and another ::type below
        template <typename F>
        auto bind(F &&f) -> lift_type<decltype(data.apply(std::forward<F>(f)))> {
            using F_result_type = decltype(data.apply(std::forward<F>(f)));
            using result_type = lift_type<F_result_type>;

            if (!set) {
                return result_type();
            }
            else return bind_impl<result_type,std::is_same<F_result_type,void>::value>::bind(data,std::forward<F>(f));
        }

        template <typename F>
        auto bind(F &&f) const -> lift_type<decltype(data.apply(std::forward<F>(f)))> {
            using F_result_type = decltype(data.apply(std::forward<F>(f)));
            using result_type = lift_type<F_result_type>;

            if (!set) {
                return result_type();
            }
            else {
                return
                    bind_impl<
                        result_type,
                        std::is_same<F_result_type,void>::value
                    >::bind(data, std::forward<F>(f));
            }
        }

        template <typename F>
        auto operator>>(F &&f) -> decltype(this->bind(std::forward<F>(f))) {
            return bind(std::forward<F>(f));
        }

        template <typename F>
        auto operator>>(F &&f) const -> decltype(this->bind(std::forward<F>(f))) {
            return bind(std::forward<F>(f));
        }

    private:
        template <typename R, bool F_void_return>
        struct bind_impl {
            template <typename DT,typename F>
            static R bind(DT &d,F &&f) {
                return R(d.apply(std::forward<F>(f)));
            }
        };

        template <typename R>
        struct bind_impl<R,true> {
            template <typename DT,typename F>
            static R bind(DT &d,F &&f) {
                d.apply(std::forward<F>(f));
                return R(true);
            }
        };
    };
}

template <typename X>
struct optional: detail::optional_base<X> {
    using base=detail::optional_base<X>;
    using base::set;
    using base::ref;
    using base::reset;
    using base::data;

    optional(): base() {}
    optional(nothing_t): base() {}

    // ... this makes it much easier to read
    template <
        typename Y = X,
        typename = typename std::enable_if<std::is_copy_constructible<Y>::value>::type
        // and this is how it would look with my style :
      //typename = std::enable_if<std::is_copy_constructible<Y>()>
    >
    optional(const X &x)
    :   base(true,x)
    {}

    // and out of curiosity, this is how it would have looked if the C++ standards
    // folks had got it right the first time

    template <typename Y=X,typename = typename std::enable_if<std::is_move_constructible<Y>::value>::type>
    optional(X &&x): base(true,std::move(x)) {}

    optional(const optional &ot): base(ot.set,ot.ref()) {}

    template <typename T>
    optional(const optional<T> &ot): base(ot.set,ot.ref()) {}

    optional(optional &&ot): base(ot.set,std::move(ot.ref())) {}

    template <typename T>
    optional(optional<T> &&ot): base(ot.set,std::move(ot.ref())) {}

    // constexpr yay!
    //template <typename Y,typename = typename std::enable_if<!detail::is_optional<Y>::value>::type>
    template <
        typename Y,
        typename = typename std::enable_if<!detail::is_optional<Y>()>::type
    >
    optional &operator=(Y &&y) {
        if (set) {
            ref()=std::forward<Y>(y);
        }
        else {
            set=true;
            data.construct(std::forward<Y>(y));
        }
        return *this;
    }

    // small style point
    //optional &operator=(const optional &o) {
    optional& operator=(const optional &o) {
        if (set) {
            if (o.set) {
                ref()=o.ref();
            }
            else {
                reset();
            }
        }
        else {
            set=o.set;
            if (set) {
                data.construct(o.ref());
            }
        }
        return *this;
    }

    // this is much clearer
    // I line the closing template > like I would a curly brace
    template <
        typename Y=X,
        typename = typename
            std::enable_if<
                std::is_move_assignable<Y>::value &&
                std::is_move_constructible<Y>::value
            >::type
    >
    optional& operator=(optional &&o) {
        // fix the {} !
        if (set) {
            if (o.set) ref()=std::move(o.ref());
            else reset();
        }
        else {
            set=o.set;
            if (set) data.construct(std::move(o.ref()));
        }
        return *this;
    }
};

template <typename X>
struct optional<X &>: detail::optional_base<X &> {
    using base=detail::optional_base<X &>;
    using base::set;
    using base::ref;
    using base::data;

    optional(): base() {}
    optional(nothing_t): base() {}
    optional(X &x): base(true,x) {}

    template <typename T>
    optional(optional<T &> &ot): base(ot.set,ot.ref()) {}

    template <typename Y,typename = typename std::enable_if<!detail::is_optional<Y>()>::type>
    optional &operator=(Y &y) {
        set=true;
        ref()=y;
        return *this;
    }

    template <typename Y>
    optional &operator=(optional<Y &> &o) {
        set=o.set;
        data.construct(o);
        return *this;
    }
};


/* special case for optional<void>, used as e.g. the result of
 * binding to a void function */

template <>
struct optional<void>: detail::optional_base<void> {
    using base=detail::optional_base<void>;
    using base::set;

    optional(): base() {}

    template <typename T>
    optional(T): base(true,true) {}

    template <typename T>
    optional(const optional<T> &o): base(o.set,true) {}
    
    template <typename T>
    optional &operator=(T) { set=true; return *this; }

    template <typename T>
    optional &operator=(const optional<T> &o) { set=o.set; return *this; }

    // override equality operators
    template <typename Y>
    bool operator==(const Y &y) const { return false; }

    bool operator==(const optional<void> &o) const {
        return (set && o.set) || (!set && !o.set);
    }
};


template <typename A,typename B>
typename std::enable_if<
    detail::is_optional<A>() || detail::is_optional<B>(),
    optional<
        typename std::common_type<
            typename detail::wrapped_type<A>,
            typename detail::wrapped_type<B>
        >::type
    >
>::type
operator|(A &&a,B &&b) {
    return a ? a : b;
}

template <typename A,typename B>
typename std::enable_if<
    detail::is_optional<A>() || detail::is_optional<B>(),
    optional<detail::wrapped_type<B>>
>::type
operator&(A&& a,B&& b) {
    using result_type=optional<detail::wrapped_type<B>>;
    return a?b:result_type();
}

inline optional<void> provided(bool condition) {
    return condition ? optional<void>(true) : optional<void>();
}

template <typename X>
optional<X> just(X &&x) {
    return optional<X>(std::forward<X>(x));
}

} // namespace util
} // namespace mc
} // namespace nest
