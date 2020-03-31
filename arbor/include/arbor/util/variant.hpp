#pragma once

// C++14 std::variant work-alike.
//
// Key differences:
//
//   * Using a type-index on operations where the type appears multiple times
//     in the variant type list is not treated as an error.
//
//   * No template constants in C++14, so `in_place_index` and `in_place_type`
//     are constexpr functions instead.
//
//   * Rather than overload `std::get` etc., uses `util::get` which wraps
//     dispatches to `variant<...>::get` (`util::get` is also defined in
//     private `util/meta.hpp` header for pairs and tuples.)
//
//   * Assignemnt from non-variant type relies upon default conversion to
//     variant type.
//
//   * Swap doesn't make nothrow guarantees.
//
//   * Unimplemented (yet): visit() with more than one variant argument;
//     monostate; comparisons; unit tests for nothrow guarantees.

#include <cstddef>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace arb {
namespace util {

struct bad_variant_access: public std::runtime_error {
    bad_variant_access(): std::runtime_error("bad variant access") {}
};

template <typename T> struct in_place_type_t {};

template <typename T>
static constexpr in_place_type_t<T> in_place_type() { return {}; }

template <std::size_t I> struct in_place_index_t {};

template <std::size_t I>
static constexpr in_place_index_t<I> in_place_index() { return {}; };

namespace detail {

template <typename... T>
struct max_sizeof: public std::integral_constant<std::size_t, 1> {};

template <typename H, typename... T>
struct max_sizeof<H, T...>: public std::integral_constant<std::size_t,
    (max_sizeof<T...>::value > sizeof(H))? max_sizeof<T...>::value: sizeof(H)> {};

template <typename... T>
struct max_alignof: public std::integral_constant<std::size_t, 1> {};

template <typename H, typename... T>
struct max_alignof<H, T...>: public std::integral_constant<std::size_t,
    (max_alignof<T...>::value > alignof(H))? max_alignof<T...>::value: alignof(H)> {};

// type_select_t<i, T0, ..., Tn> gives type Ti.

template <std::size_t I, typename... T> struct type_select;

template <typename X, typename... T>
struct type_select<0, X, T...> { using type = X; };

template <std::size_t I, typename X, typename... T>
struct type_select<I, X, T...> { using type = typename type_select<I-1, T...>::type; };

template <std::size_t I>
struct type_select<I> { using type = void; };

template <std::size_t I, typename... T>
using type_select_t = typename type_select<I, T...>::type;

// type_index<T, T0, ..., Tn>::value gives i such that T is Ti, or else -1.

template <std::size_t I, typename X, typename... T>
struct type_index_impl: std::integral_constant<std::size_t, std::size_t(-1)> {};

template <std::size_t I, typename X, typename... T>
struct type_index_impl<I, X, X, T...>: std::integral_constant<std::size_t, I> {};

template <std::size_t I, typename X, typename Y, typename... T>
struct type_index_impl<I, X, Y, T...>: type_index_impl<I+1, X, T...> {};

template <typename X, typename... T>
using type_index = std::integral_constant<std::size_t, type_index_impl<0, X, T...>::value>;

// Build overload set for implicit construction from type list.

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <std::size_t, typename... T>
struct variant_implicit_ctor_index_impl;

template <std::size_t I>
struct variant_implicit_ctor_index_impl<I> {
    static std::integral_constant<std::size_t, std::size_t(-1)> index(...);
};

template <std::size_t I, typename X, typename... T>
struct variant_implicit_ctor_index_impl<I, X, T...>: variant_implicit_ctor_index_impl<I+1, T...> {
    using variant_implicit_ctor_index_impl<I+1, T...>::index;

    template <typename X_nocv = std::remove_cv_t<X>,
              typename = std::enable_if_t<!std::is_same<bool, X_nocv>::value>>
    static std::integral_constant<std::size_t, I> index(X);

    template <typename A,
              typename X_nocv = std::remove_cv_t<X>,
              typename = std::enable_if_t<std::is_same<bool, X_nocv>::value>,
              typename A_nocvref = remove_cvref_t<A>,
              typename = std::enable_if_t<std::is_same<bool, A_nocvref>::value>>
    static std::integral_constant<std::size_t, I> index(A&& a);
};

template <typename X, typename... T>
struct variant_implicit_ctor_index:
    decltype(variant_implicit_ctor_index_impl<0, T...>::index(std::declval<X>())) {};

// Test for in-place types

template <typename X> struct is_in_place_impl: std::false_type {};
template <typename T> struct is_in_place_impl<in_place_type_t<T>>: std::true_type {};
template <std::size_t I> struct is_in_place_impl<in_place_index_t<I>>: std::true_type {};

template <typename X> using is_in_place = is_in_place_impl<std::decay_t<X>>;

// Variadic tests for nothrow.

template <typename... T> struct are_nothrow_move_constructible;
template <> struct are_nothrow_move_constructible<>: std::true_type {};
template <typename H, typename... T>
struct are_nothrow_move_constructible<H, T...>:
    std::conditional_t<std::is_nothrow_move_constructible<H>::value,
                       are_nothrow_move_constructible<T...>, std::false_type> {};

template <typename... T> struct are_nothrow_copy_constructible;
template <> struct are_nothrow_copy_constructible<>: std::true_type {};
template <typename H, typename... T>
struct are_nothrow_copy_constructible<H, T...>:
    std::conditional_t<std::is_nothrow_copy_constructible<H>::value,
                       are_nothrow_copy_constructible<T...>, std::false_type> {};

template <typename... T> struct any_reference;
template <> struct any_reference<>: std::false_type {};
template <typename H, typename... T>
struct any_reference<H, T...>:
    std::conditional_t<std::is_reference<H>::value, std::true_type, any_reference<T...>> {};

// Copy and move ctor and assignment implementations.

template <typename... T>
struct variant_dynamic_impl;

template <>
struct variant_dynamic_impl<> {
    static void copy(std::size_t i, char* data, const char* from) {
        if (i!=std::size_t(-1)) throw bad_variant_access{};
    }

    static void move(std::size_t i, char* data, const char* from) {
        if (i!=std::size_t(-1)) throw bad_variant_access{};
    }

    static void assign(std::size_t i, char* data, const char* from) {
        if (i!=std::size_t(-1)) throw bad_variant_access{};
    }

    static void move_assign(std::size_t i, char* data, const char* from) {
        if (i!=std::size_t(-1)) throw bad_variant_access{};
    }

    static void swap(std::size_t i, char* data1, char* data2) {
        if (i!=std::size_t(-1)) throw bad_variant_access{};
    }

    static void destroy(std::size_t i, char* data) {}
};

template <typename H, typename... T>
struct variant_dynamic_impl<H, T...> {
    static void copy(std::size_t i, char* data, const char* from) {
        if (i==0) {
            new(reinterpret_cast<H*>(data)) H(*reinterpret_cast<const H*>(from));
        }
        else {
            variant_dynamic_impl<T...>::copy(i-1, data, from);
        }
    }

    static void move(std::size_t i, char* data, char* from) {
        if (i==0) {
            new(reinterpret_cast<H*>(data)) H(std::move(*reinterpret_cast<H*>(from)));
        }
        else {
            variant_dynamic_impl<T...>::move(i-1, data, from);
        }
    }

    static void assign(std::size_t i, char* data, const char* from) {
        if (i==0) {
            *reinterpret_cast<H*>(data) = *reinterpret_cast<const H*>(from);
        }
        else {
            variant_dynamic_impl<T...>::assign(i-1, data, from);
        }
        if (i!=std::size_t(-1)) throw bad_variant_access{};
    }

    static void move_assign(std::size_t i, char* data, const char* from) {
        if (i==0) {
            *reinterpret_cast<H*>(data) = std::move(*reinterpret_cast<const H*>(from));
        }
        else {
            variant_dynamic_impl<T...>::move_assign(i-1, data, from);
        }
    }

    static void swap(std::size_t i, char* data1, char* data2) {
        using std::swap;
        if (i==0) {
            swap(*reinterpret_cast<H*>(data1), *reinterpret_cast<H*>(data2));
        }
        else {
            variant_dynamic_impl<T...>::swap(i-1, data1, data2);
        }
    }

    static void destroy(std::size_t i, char* data) {
        if (i==0) {
            reinterpret_cast<H*>(data)->~H();
        }
        else {
            variant_dynamic_impl<T...>::destroy(i-1, data);
        }
    }
};

template <typename... T>
struct variant {
    static_assert(!any_reference<T...>::value, "variant must have no reference alternative");
    alignas(max_alignof<T...>::value) char data[max_sizeof<T...>::value];

    template <typename X> X* data_ptr() noexcept { return reinterpret_cast<X*>(&data); }
    template <typename X> const X* data_ptr() const noexcept { return reinterpret_cast<const X*>(&data); }

    std::size_t which_ = -1;
    static constexpr std::size_t npos = -1;

    // Explict construction by index.

    template <std::size_t I, typename... A, typename = std::enable_if_t<(I<sizeof...(T))>>
    explicit variant(in_place_index_t<I>, A&&... a): which_(I)
    {
        using X = type_select_t<I, T...>;
        new(data_ptr<X>()) X(std::forward<A>(a)...);
    }

    template <std::size_t I, typename U, typename... A, typename = std::enable_if_t<(I<sizeof...(T))>>
    explicit variant(in_place_index_t<I>, std::initializer_list<U> il, A&&... a): which_(I)
    {
        using X = type_select_t<I, T...>;
        new(data_ptr<X>()) X(il, std::forward<A>(a)...);
    }

    // Explicit construction by type.

    template <typename X, typename... A, std::size_t I = type_index<X, T...>::value>
    explicit variant(in_place_type_t<X>, A&&... a):
        variant(in_place_index_t<I>{}, std::forward<A>(a)...) {}

    template <typename X, typename U, typename... A, std::size_t I = type_index<X, T...>::value>
    explicit variant(in_place_type_t<X>, std::initializer_list<U> il, A&&... a):
        variant(in_place_index_t<I>{}, il, std::forward<A>(a)...) {}

    // Implicit construction from argument.

    template <typename X,
              typename = std::enable_if_t<!std::is_same<variant, std::decay_t<X>>::value>,
              typename = std::enable_if_t<!is_in_place<X>::value>,
              typename index = variant_implicit_ctor_index<X, T...>>
    variant(X&& x):
        variant(in_place_index<index::value>(), std::forward<X>(x)) {}

    // Default constructible if first type is.

    template <typename X = type_select_t<0, T...>,
        typename = std::enable_if_t<std::is_default_constructible<X>::value>>
    variant() noexcept(std::is_nothrow_default_constructible<X>::value): which_(0) {
        new(data_ptr<X>()) X;
    }

    // Copy construction.

    variant(const variant& x)
        noexcept(are_nothrow_copy_constructible<T...>::value): which_(x.which_)
    {
        variant_dynamic_impl<T...>::copy(which_, data, x.data);
    }

    // Move construction.

    variant(variant&& x)
        noexcept(are_nothrow_move_constructible<T...>::value): which_(x.which_)
    {
        variant_dynamic_impl<T...>::move(which_, data, x.data);
    }

    // Copy assignment.

    variant& operator=(const variant& x) {
        if (which_!=x.which_) {
            variant_dynamic_impl<T...>::destroy(which_, data);
            which_ = npos;
            if (x.which_!=npos) {
                variant_dynamic_impl<T...>::copy(x.which_, data, x.data);
                which_ = x.which_;
            }
        }
        else {
            which_ = npos;
            if (x.which_!=npos) {
                variant_dynamic_impl<T...>::assign(x.which_, data, x.data);
                which_ = x.which_;
            }
        }
        return *this;
    }

    // Move assignment.

    variant& operator=(variant&& x) {
        if (which_!=x.which_) {
            variant_dynamic_impl<T...>::destroy(which_, data);
            which_ = npos;
            if (x.which_!=npos) {
                variant_dynamic_impl<T...>::move(x.which_, data, x.data);
                which_ = x.which_;
            }
        }
        else {
            which_ = npos;
            if (x.which_!=npos) {
                variant_dynamic_impl<T...>::move_assign(x.which_, data, x.data);
                which_ = x.which_;
            }
        }
        return *this;
    }

    // In place construction.

    template <std::size_t I,
              typename... Args,
              typename = std::enable_if_t<(I<sizeof...(T))>,
              typename X = type_select_t<I, T...>,
              typename = std::enable_if_t<std::is_constructible<X, Args...>::value>>
    X& emplace(Args&&... args) {
        if (which_!=npos) {
            variant_dynamic_impl<T...>::destroy(which_, data);
            which_ = npos;
        }
        new(data_ptr<X>()) X(std::forward<Args>(args)...);
        return *data_ptr<X>();
    }

    template <std::size_t I,
              typename U,
              typename... Args,
              typename = std::enable_if_t<(I<sizeof...(T))>,
              typename X = type_select_t<I, T...>,
              typename = std::enable_if_t<std::is_constructible<X, std::initializer_list<U>, Args...>::value>>
    X& emplace(std::initializer_list<U> il, Args&&... args) {
        if (which_!=npos) {
            variant_dynamic_impl<T...>::destroy(which_, data);
            which_ = npos;
        }
        new(data_ptr<X>()) X(il, std::forward<Args>(args)...);
        which_ = I;
        return *data_ptr<X>();
    }

    template <typename X,
              typename... Args,
              std::size_t I = type_index<X, T...>::value>
    X& emplace(Args&&... args) {
        return emplace<I>(std::forward<Args>(args)...);
    }

    template <typename X,
              typename U,
              typename... Args,
              std::size_t I = type_index<X, T...>::value>
    X& emplace(std::initializer_list<U> il, Args&&... args) {
        return emplace<I>(il, std::forward<Args>(args)...);
    }

    // Swap.

    void swap(variant& rhs) {
        if (which_==rhs.which_) {
            if (which_!=npos) {
                variant_dynamic_impl<T...>::swap(which_, data, rhs.data);
            }
        }
        else {
            variant tmp(std::move(rhs));
            rhs = std::move(*this);
            *this = std::move(tmp);
        }
    }

    // Queries.

    std::size_t index() const { return which_; }

    bool valueless_by_exception() const { return which_==npos; }

    // Pointer access (does not throw).

    template <std::size_t I, typename = std::enable_if_t<(I<sizeof...(T))>, typename X = type_select_t<I, T...>>
    X* get_if() noexcept { return which_==I? data_ptr<X>(): nullptr; }

    template <typename X, std::size_t I = type_index<X, T...>::value>
    auto get_if() noexcept { return get_if<I>(); }

    template <std::size_t I, typename = std::enable_if_t<(I<sizeof...(T))>, typename X = type_select_t<I, T...>>
    const X* get_if() const noexcept { return which_==I? data_ptr<>(): nullptr; }

    template <typename X, std::size_t I = type_index<X, T...>::value>
    auto get_if() const noexcept { return get_if<I>(); }

    // Reference access (throws).

    template <std::size_t I, typename = std::enable_if_t<(I<sizeof...(T))>>
    auto& get() & {
        if (auto* p = get_if<I>()) return *p;
        else throw bad_variant_access{};
    }

    template <std::size_t I, typename = std::enable_if_t<(I<sizeof...(T))>>
    auto& get() const & {
        if (auto* p = get_if<I>()) return *p;
        else throw bad_variant_access{};
    }

    template <std::size_t I, typename = std::enable_if_t<(I<sizeof...(T))>>
    auto&& get() && {
        if (auto* p = get_if<I>()) return std::move(*p);
        else throw bad_variant_access{};
    }

    template <std::size_t I, typename = std::enable_if_t<(I<sizeof...(T))>>
    auto&& get() const && {
        if (auto* p = get_if<I>()) return std::move(*p);
        else throw bad_variant_access{};
    }

    template <typename X, std::size_t I = type_index<X, T...>::value>
    decltype(auto) get() { return get<I>(); }

    template <typename X, std::size_t I = type_index<X, T...>::value>
    decltype(auto) get() const { return get<I>(); }
};

template <std::size_t I, std::size_t N>
struct variant_visit {
    template <typename R, typename Visitor, typename Variant>
    static R visit(std::size_t i, Visitor&& f, Variant&& v) {
        if (i==I) {
            return static_cast<R>(std::forward<Visitor>(f)(std::forward<Variant>(v).template get<I>()));
        }
        else {
            return variant_visit<I+1, N>::template visit<R>(i, std::forward<Visitor>(f), std::forward<Variant>(v));
        }
    }
};

template <std::size_t I>
struct variant_visit<I, I> {
    template <typename R, typename Visitor, typename Variant>
    static R visit(std::size_t i, Visitor&& f, Variant&& v) {
        throw bad_variant_access{}; // Actually, should never get here.
    }
};

template <typename X> struct variant_size_impl;
template <typename... T>
struct variant_size_impl<variant<T...>>: std::integral_constant<std::size_t, sizeof...(T)> {};

template <std::size_t I, typename T> struct variant_alternative;

template <std::size_t I, typename... T>
struct variant_alternative<I, variant<T...>> { using type = type_select_t<I, T...>; };

template <std::size_t I, typename... T>
struct variant_alternative<I, const variant<T...>> { using type = std::add_const_t<type_select_t<I, T...>>; };

template <typename Visitor, typename... Variant>
using visit_return_t = decltype(std::declval<Visitor>()(std::declval<typename variant_alternative<0, std::remove_volatile_t<std::remove_reference_t<Variant>>>::type>()...));

} // namespace detail

template <typename... T>
using variant = detail::variant<T...>;

template <typename X>
using variant_size = detail::variant_size_impl<std::remove_cv_t<std::remove_reference_t<X>>>;

template <std::size_t I, typename V>
using variant_alternative_t = typename detail::variant_alternative<I, V>::type;

// util:: variants of std::get

template <typename X, typename... T>
decltype(auto) get(variant<T...>& v) { return v.template get<X>(); }

template <typename X, typename... T>
decltype(auto) get(const variant<T...>& v) { return v.template get<X>(); }

template <typename X, typename... T>
decltype(auto) get(variant<T...>&& v) { return std::move(v).template get<X>(); }

template <typename X, typename... T>
decltype(auto) get(const variant<T...>&& v) { return std::move(v).template get<X>(); }

template <std::size_t I, typename... T>
decltype(auto) get(variant<T...>& v) { return v.template get<I>(); }

template <std::size_t I, typename... T>
decltype(auto) get(const variant<T...>& v) { return v.template get<I>(); }

template <std::size_t I, typename... T>
decltype(auto) get(variant<T...>&& v) { return std::move(v).template get<I>(); }

template <std::size_t I, typename... T>
decltype(auto) get(const variant<T...>&& v) { return std::move(v).template get<I>(); }

// util:: variants of std::get_if

template <typename X, typename... T>
decltype(auto) get_if(variant<T...>& v) noexcept { return v.template get_if<X>(); }

template <typename X, typename... T>
decltype(auto) get_if(const variant<T...>& v) noexcept { return v.template get_if<X>(); }

template <std::size_t I, typename... T>
decltype(auto) get_if(variant<T...>& v) noexcept { return v.template get_if<I>(); }

template <std::size_t I, typename... T>
decltype(auto) get_if(const variant<T...>& v) noexcept { return v.template get_if<I>(); }

// One-argument visitor

template <typename Visitor, typename Variant>
decltype(auto) visit(Visitor&& f, Variant&& v) {
    using R = detail::visit_return_t<Visitor&&, Variant&&>;

    if (v.valueless_by_exception()) throw bad_variant_access{};
    std::size_t i = v.index();
    return static_cast<R>(detail::variant_visit<0, variant_size<Variant>::value>::template visit<R>(i,
        std::forward<Visitor>(f), std::forward<Variant>(v)));
}

template <typename R, typename Visitor, typename Variant>
R visit(Visitor&& f, Variant&& v) {
    if (v.valueless_by_exception()) throw bad_variant_access{};
    std::size_t i = v.index();
    return static_cast<R>(detail::variant_visit<0, variant_size<Variant>::value>::template visit<R>(i,
        std::forward<Visitor>(f), std::forward<Variant>(v)));
}

// Not implementing multi-argument visitor yet!
// (If we ever have a use case...)

} // namespace util
} // namespace arb

namespace std {

// Unambitious hash:
template <typename... T>
struct hash<::arb::util::variant<T...>> {
    std::size_t operator()(const ::arb::util::variant<T...>& v) {
        return v.index() ^
            visit([](const auto& a) { return std::hash<std::remove_cv_t<decltype(a)>>{}(a); }, v);
    }
};

// Still haven't really determined if it is okay to have a variant<>, but if we do allow it...
template <>
struct hash<::arb::util::variant<>> {
    std::size_t operator()(const ::arb::util::variant<>& v) { return 0u; };
};

// std::swap specialization.
template <typename... T>
void swap(::arb::util::variant<T...>& v1, ::arb::util::variant<T...>& v2) {
    v1.swap(v2);
}
} // namespace std
