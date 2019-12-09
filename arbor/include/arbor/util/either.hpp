#pragma once

/*
 * A type-safe discriminated union of two members.
 *
 * Returns true in a bool context if the first of the two types holds a value.
 */

#include <cstdlib>
#include <type_traits>
#include <stdexcept>
#include <utility>

#include <arbor/util/uninitialized.hpp>

namespace arb {
namespace util {

struct either_invalid_access: std::runtime_error {
    explicit either_invalid_access(const std::string& what_str)
        : std::runtime_error(what_str)
    {}

    either_invalid_access()
        : std::runtime_error("access of unconstructed value in either")
    {}
};

namespace detail {
    template <typename A, typename B>
    struct either_data {
        union {
            uninitialized<A> ua;
            uninitialized<B> ub;
        };

        either_data() = default;

        either_data(const either_data&) = delete;
        either_data(either_data&&) = delete;
        either_data& operator=(const either_data&) = delete;
        either_data& operator=(either_data&&) = delete;
    };

    template <std::size_t, typename A, typename B> struct either_select;

    template <typename A, typename B>
    struct either_select<0, A, B> {
        using type = uninitialized<A>;
        static type& field(either_data<A, B>& data) { return data.ua; }
        static const type& field(const either_data<A, B>& data) { return data.ua; }
    };

    template <typename A, typename B>
    struct either_select<1, A, B> {
        using type = uninitialized<B>;
        static type& field(either_data<A, B>& data) { return data.ub; }
        static const type& field(const either_data<A, B>& data) { return data.ub; }
    };

    template <std::size_t I, typename A, typename B>
    struct either_get: either_select<I, A, B> {
        using typename either_select<I, A, B>::type;
        using either_select<I, A, B>::field;

        static typename type::reference unsafe_get(either_data<A, B>& data) {
            return field(data).ref();
        }

        static typename type::const_reference unsafe_get(const either_data<A, B>& data) {
            return field(data).cref();
        }

        static typename type::reference unsafe_get(char which, either_data<A, B>& data) {
            if (I!=which) {
                throw either_invalid_access();
            }
            return field(data).ref();
        }

        static typename type::const_reference unsafe_get(char which, const either_data<A, B>& data) {
            if (I!=which) {
                throw either_invalid_access();
            }
            return field(data).cref();
        }

        static typename type::pointer ptr(char which, either_data<A, B>& data) {
            return I==which? field(data).ptr(): nullptr;
        }

        static typename type::const_pointer ptr(char which, const either_data<A, B>& data) {
            return I==which? field(data).cptr(): nullptr;
        }
    };
} // namespace detail

constexpr std::size_t variant_npos = static_cast<std::size_t>(-1); // emulating C++17 variant type

template <typename A, typename B>
class either: public detail::either_data<A, B> {
    using base = detail::either_data<A, B>;
    using base::ua;
    using base::ub;

    template <std::size_t I>
    using getter = detail::either_get<I, A, B>;

    unsigned char which;

public:
    // default ctor if A is default-constructible or A is not and B is.
    template <
        typename A_ = A,
        bool a_ = std::is_default_constructible<A_>::value,
        bool b_ = std::is_default_constructible<B>::value,
        typename = std::enable_if_t<a_ || (!a_ && b_)>,
        std::size_t w_ = a_? 0: 1
    >
    either() noexcept(std::is_nothrow_default_constructible<typename getter<w_>::type>::value):
        which(w_)
    {
        getter<w_>::field(*this).construct();
    }

    // implicit constructors from A and B values by copy or move
    either(const A& a) noexcept(std::is_nothrow_copy_constructible<A>::value): which(0) {
        getter<0>::field(*this).construct(a);
    }

    template <
        typename B_ = B,
        typename = std::enable_if_t<!std::is_same<A, B_>::value>
    >
    either(const B& b) noexcept(std::is_nothrow_copy_constructible<B>::value): which(1) {
        getter<1>::field(*this).construct(b);
    }

    either(A&& a) noexcept(std::is_nothrow_move_constructible<A>::value): which(0) {
        getter<0>::field(*this).construct(std::move(a));
    }

    template <
        typename B_ = B,
        typename = std::enable_if_t<!std::is_same<A, B_>::value>
    >
    either(B&& b) noexcept(std::is_nothrow_move_constructible<B>::value): which(1) {
        getter<1>::field(*this).construct(std::move(b));
    }

    // copy constructor
    either(const either& x)
        noexcept(std::is_nothrow_copy_constructible<A>::value &&
            std::is_nothrow_copy_constructible<B>::value):
        which(x.which)
    {
        if (which==0) {
            getter<0>::field(*this).construct(x.unsafe_get<0>());
        }
        else if (which==1) {
            getter<1>::field(*this).construct(x.unsafe_get<1>());
        }
    }

    // move constructor
    either(either&& x)
        noexcept(std::is_nothrow_move_constructible<A>::value &&
            std::is_nothrow_move_constructible<B>::value):
        which(x.which)
    {
        if (which==0) {
            getter<0>::field(*this).construct(std::move(x.unsafe_get<0>()));
        }
        else if (which==1) {
            getter<1>::field(*this).construct(std::move(x.unsafe_get<1>()));
        }
    }

    // copy assignment
    either& operator=(const either& x) {
        if (this==&x) {
            return *this;
        }

        switch (which) {
        case 0:
            if (x.which==0) {
                unsafe_get<0>() = x.unsafe_get<0>();
            }
            else {
                if (x.which==1) {
                    B b_tmp(x.unsafe_get<1>());
                    getter<0>::field(*this).destruct();
                    which = (unsigned char)variant_npos;
                    getter<1>::field(*this).construct(std::move(b_tmp));
                    which = 1;
                }
                else {
                    getter<0>::field(*this).destruct();
                    which = (unsigned char)variant_npos;
                }
            }
            break;
        case 1:
            if (x.which==1) {
                unsafe_get<1>() = x.unsafe_get<1>();
            }
            else {
                if (x.which==0) {
                    A a_tmp(x.unsafe_get<0>());
                    getter<1>::field(*this).destruct();
                    which = (unsigned char)variant_npos;
                    getter<0>::field(*this).construct(std::move(a_tmp));
                    which = 0;
                }
                else {
                    getter<1>::field(*this).destruct();
                    which = (unsigned char)variant_npos;
                }
            }
            break;
        default: // variant_npos
            if (x.which==0) {
                getter<0>::field(*this).construct(x.unsafe_get<0>());
            }
            else if (x.which==1) {
                getter<1>::field(*this).construct(x.unsafe_get<1>());
            }
            break;
        }
        return *this;
    }

    // move assignment
    either& operator=(either&& x) {
        if (this==&x) {
            return *this;
        }

        switch (which) {
        case 0:
            if (x.which==0) {
                unsafe_get<0>() = std::move(x.unsafe_get<0>());
            }
            else {
                which = (unsigned char)variant_npos;
                getter<0>::field(*this).destruct();
                if (x.which==1) {
                    getter<1>::field(*this).construct(std::move(x.unsafe_get<1>()));
                    which = 1;
                }
            }
            break;
        case 1:
            if (x.which==1) {
                unsafe_get<1>() = std::move(x.unsafe_get<1>());
            }
            else {
                which = (unsigned char)variant_npos;
                getter<1>::field(*this).destruct();
                if (x.which==0) {
                    getter<0>::field(*this).construct(std::move(x.unsafe_get<0>()));
                    which = 0;
                }
            }
            break;
        default: // variant_npos
            if (x.which==0) {
                getter<0>::field(*this).construct(std::move(x.unsafe_get<0>()));
            }
            else if (x.which==1) {
                getter<1>::field(*this).construct(std::move(x.unsafe_get<1>()));
            }
            break;
        }
        return *this;
    }

    // unchecked element access
    template <std::size_t I>
    typename getter<I>::type::reference unsafe_get() {
        return getter<I>::unsafe_get(*this);
    }

    template <std::size_t I>
    typename getter<I>::type::const_reference unsafe_get() const {
        return getter<I>::unsafe_get(*this);
    }

    // checked element access
    template <std::size_t I>
    typename getter<I>::type::reference get() {
        return getter<I>::unsafe_get(which, *this);
    }

    template <std::size_t I>
    typename getter<I>::type::const_reference get() const {
        return getter<I>::unsafe_get(which, *this);
    }

    // convenience getter aliases
    typename getter<0>::type::reference first() { return get<0>(); }
    typename getter<0>::type::const_reference first() const { return get<0>(); }

    typename getter<1>::type::reference second() { return get<1>(); }
    typename getter<1>::type::const_reference second() const  { return get<1>(); }

    // pointer to element access: return nullptr if it does not hold this item
    template <std::size_t I>
    auto ptr() {
        return getter<I>::ptr(which, *this);
    }

    template <std::size_t I>
    auto ptr() const {
        return getter<I>::ptr(which, *this);
    }

    // true in bool context if holds first alternative
    constexpr operator bool() const { return which==0; }

    constexpr bool valueless_by_exception() const noexcept {
        return which==(unsigned char)variant_npos;
    }

    constexpr std::size_t index() const noexcept {
        return which;
    }

    ~either() {
        if (which==0) {
            getter<0>::field(*this).destruct();
        }
        else if (which==1) {
            getter<1>::field(*this).destruct();
        }
    }

    // comparison operators follow C++17 variant semantics
    bool operator==(const either& x) const {
        return index()==x.index() &&
           index()==0? unsafe_get<0>()==x.unsafe_get<0>():
           index()==1? unsafe_get<1>()==x.unsafe_get<1>():
           true;
    }

    bool operator!=(const either& x) const {
        return index()!=x.index() ||
           index()==0? unsafe_get<0>()!=x.unsafe_get<0>():
           index()==1? unsafe_get<1>()!=x.unsafe_get<1>():
           false;
    }

    bool operator<(const either& x) const {
        return !x.valueless_by_exception() &&
           index()==0? (x.index()==1 || unsafe_get<0>()<x.unsafe_get<0>()):
           index()==1? (x.index()!=0 && unsafe_get<1>()<x.unsafe_get<1>()):
           true;
    }

    bool operator>=(const either& x) const {
        return x.valueless_by_exception() ||
           index()==0? (x.index()!=1 && unsafe_get<0>()>=x.unsafe_get<0>()):
           index()==1? (x.index()==0 || unsafe_get<1>()>=x.unsafe_get<1>()):
           false;
    }

    bool operator<=(const either& x) const {
        return valueless_by_exception() ||
           x.index()==0? (index()!=1 && unsafe_get<0>()<=x.unsafe_get<0>()):
           x.index()==1? (index()==0 || unsafe_get<1>()<=x.unsafe_get<1>()):
           false;
    }

    bool operator>(const either& x) const {
        return !valueless_by_exception() &&
           x.index()==0? (index()==1 || unsafe_get<0>()>x.unsafe_get<0>()):
           x.index()==1? (index()!=0 && unsafe_get<1>()>x.unsafe_get<1>()):
           true;
    }
};

} // namespace util
} // namespace arb
