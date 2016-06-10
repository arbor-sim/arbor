/* Represent a possibly-uninitialized value, reference or void.
 *
 * The uninitialized<X> structure holds space for an item of
 * type X, leaving its construction or destruction to the user.
 * 
 * Specialisations for reference types X & and for the void type
 * allow for the handling of non-value types in a uniform manner.
 */

#ifndef UTIL_UNINITIALIZED_H_
#define UTIL_UNINITIALIZED_H_

namespace nest {
namespace mc {
namespace util {

/* Maintains storage for a value of type X, with explicit
 * construction and destruction.
 */
template <typename X>
struct uninitialized {
private:
    typename std::aligned_storage<sizeof(X),alignof(X)>::type data;

public:
    typedef X *pointer_type;
    typedef const X *const_pointer_type;
    typedef X &reference_type;
    typedef const X &const_reference_type;

    pointer_type ptr() { return reinterpret_cast<X *>(&data); }
    const_pointer_type cptr() const { return reinterpret_cast<const X *>(&data); }

    reference_type ref() { return *reinterpret_cast<X *>(&data); }
    const_reference_type cref() const { return *reinterpret_cast<const X *>(&data); }

    // Copy construct the value.
    template <typename Y=X,
              typename =typename std::enable_if<std::is_copy_constructible<Y>::value>::type>
    void construct(const X &x) { new(&data) X(x); }

    // General constructor for X, forwarding arguments.
    template <typename... Y,
              typename =typename std::enable_if<std::is_constructible<X,Y...>::value>::type>
    void construct(Y&& ...args) { new(&data) X(std::forward<Y>(args)...); }

    void destruct() { ptr()->~X(); }

    // Apply the one-parameter functor F to the value by reference.
    template <typename F>
    typename std::result_of<F(reference_type)>::type apply(F &&f) { return f(ref()); }

    // Apply the one-parameter functor F to the value by const reference.
    template <typename F>
    typename std::result_of<F(const_reference_type)>::type apply(F &&f) const { return f(cref()); }
};

/* Maintains storage for a pointer of type X, representing
 * a possibly uninitialized reference.
 */
template <typename X>
struct uninitialized<X&> {
private:
    X *data;

public:
    typedef X *pointer_type;
    typedef const X *const_pointer_type;
    typedef X &reference_type;
    typedef const X &const_reference_type;

    pointer_type ptr() { return data; }
    const_pointer_type cptr() const { return data; }

    reference_type ref() { return *data; }
    const_reference_type cref() const { return *data; }

    void construct(X &x) { data=&x; }
    void destruct() {}

    // Apply the one-parameter functor F to the value by reference.
    template <typename F>
    typename std::result_of<F(reference_type)>::type apply(F &&f) { return f(ref()); }
    // Apply the one-parameter functor F to the value by const reference.
    template <typename F>
    typename std::result_of<F(const_reference_type)>::type apply(F &&f) const { return f(cref()); }
};

/* Wrap a void type in an uninitialized template.
 * 
 * Allows the use of uninitialized<X> for void X, for generic applications.
 */
template <>
struct uninitialized<void> {
    typedef void *pointer_type;
    typedef const void *const_pointer_type;
    typedef void reference_type;
    typedef void const_reference_type;

    pointer_type ptr() { return nullptr; }
    const_pointer_type cptr() const { return nullptr; }

    reference_type ref() {}
    const_reference_type cref() const {}

    // No operation.
    void construct(...) {}
    // No operation.
    void destruct() {}

    // Equivalent to f()
    template <typename F>
    typename std::result_of<F()>::type apply(F &&f) const { return f(); }
};

template <typename...>
struct uninitialized_can_construct: std::false_type {};

template <typename X,typename... Y>
struct uninitialized_can_construct<X,Y...>: std::integral_constant<bool,std::is_constructible<X,Y...>::value> {};

template <typename X,typename Y>
struct uninitialized_can_construct<X &,Y>: std::integral_constant<bool,std::is_convertible<X &,Y>::value> {};

template <typename... Y>
struct uninitialized_can_construct<void,Y...>: std::true_type {};

}}} // namespace nest::mc::util

#endif // ndef UTIL_UNINITIALIZED_H_

