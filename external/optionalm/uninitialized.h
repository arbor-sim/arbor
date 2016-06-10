/*! \file uninitialized.h
 *  \brief Represent a possibly-uninitialized value, reference or void.
 *
 *  The \c uninitialized\<X\> structure holds space for an item of
 *  type \c X, leaving its construction or destruction to the user.
 */

#ifndef HF_UNINITIALIZED_H_
#define HF_UNINITIALIZED_H_

namespace hf {
namespace optionalm {

/*! \defgroup uninitialized The uninitialized<X> classes.
 *  \{
 * 
 *  The \c uninitialized\<X\> structure holds space for an item of
 *  type \c X, which can then be explicitly constructed or
 *  deconstructed through the \c construct() and \c destruct()
 *  member functions.
 *
 *  Specialisations for reference types <tt>X &</tt> and for the
 *  \c void type allow handling non-value types in a uniform
 *  manner.
 */

/*! \brief Maintains storage for a value of type X, with explicit
 *  construction and destruction.
 *
 *  \tparam X
 *      The wrapped value type.
 */
template <typename X>
struct uninitialized {
private:
    typename std::aligned_storage<sizeof(X),alignof(X)>::type data;

public:
    //! &nbsp;
    typedef X *pointer_type;
    //! &nbsp;
    typedef const X *const_pointer_type;
    //! &nbsp;
    typedef X &reference_type;
    //! &nbsp;
    typedef const X &const_reference_type;

    //! Return a pointer to the value.
    pointer_type ptr() { return reinterpret_cast<X *>(&data); }
    //! Return a const pointer to the value.
    const_pointer_type cptr() const { return reinterpret_cast<const X *>(&data); }

    //! Return a reference to the value.
    reference_type ref() { return *reinterpret_cast<X *>(&data); }
    //! Return a const reference to the value.
    const_reference_type cref() const { return *reinterpret_cast<const X *>(&data); }

    //! Copy construct the value.
    template <typename Y=X,typename =typename std::enable_if<std::is_copy_constructible<Y>::value>::type>
    void construct(const X &x) { new(&data) X(x); }

    //! General constructor
    template <typename... Y,typename =typename std::enable_if<std::is_constructible<X,Y...>::value>::type>
    void construct(Y&& ...args) { new(&data) X(std::forward<Y>(args)...); }

    //! Call the destructor of the value.
    void destruct() { ptr()->~X(); }

    //! Apply the one-parameter functor F to the value by reference.
    template <typename F>
    typename std::result_of<F(reference_type)>::type apply(F &&f) { return f(ref()); }
    //! Apply the one-parameter functor F to the value by const reference.
    template <typename F>
    typename std::result_of<F(const_reference_type)>::type apply(F &&f) const { return f(cref()); }
};

/*! \brief Maintains storage for a pointer of type X, representing
 *  a possibly uninitialized reference.
 *
 *  \tparam X& 
 *      Wrapped reference type.
 */
template <typename X>
struct uninitialized<X&> {
private:
    X *data;

public:
    //! &nbsp;
    typedef X *pointer_type;
    //! &nbsp;
    typedef const X *const_pointer_type;
    //! &nbsp;
    typedef X &reference_type;
    //! &nbsp;
    typedef const X &const_reference_type;

    //! Return a pointer to the value.
    pointer_type ptr() { return data; }
    //! Return a const pointer to the value.
    const_pointer_type cptr() const { return data; }

    //! Return a reference to the value.
    reference_type ref() { return *data; }
    //! Return a const reference to the value.
    const_reference_type cref() const { return *data; }

    //! Set the reference data.
    void construct(X &x) { data=&x; }
    //! Destruct is a NOP for reference data.
    void destruct() {}

    //! Apply the one-parameter functor F to the value by reference.
    template <typename F>
    typename std::result_of<F(reference_type)>::type apply(F &&f) { return f(ref()); }
    //! Apply the one-parameter functor F to the value by const reference.
    template <typename F>
    typename std::result_of<F(const_reference_type)>::type apply(F &&f) const { return f(cref()); }
};

/*! \brief Wrap a void type in an uninitialized template.
 * 
 *  Allows the use of <tt>uninitialized\<X\></tt> for void \c X, for generic
 *  applications.
 */

template <>
struct uninitialized<void> {
    //! &nbsp;
    typedef void *pointer_type;
    //! &nbsp;
    typedef const void *const_pointer_type;
    //! &nbsp;
    typedef void reference_type;
    //! &nbsp;
    typedef void const_reference_type;

    //! &nbsp;
    pointer_type ptr() { return nullptr; }
    //! &nbsp;
    const_pointer_type cptr() const { return nullptr; }

    //! &nbsp;
    reference_type ref() {}
    //! &nbsp;
    const_reference_type cref() const {}

    //! No operation.
    void construct(...) {}
    //! No operation.
    void destruct() {}

    //! Equivalent to <tt>f()</tt>
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

/*! \} */

}} // namespace hf::optionalm

#endif // ndef HF_UNINITIALIZED_H_

