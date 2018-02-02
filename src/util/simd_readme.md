# SIMD classes for Arbor

The purpose of the SIMD classes is to abstract and consolidate the use of
compiler intrinsics for the manipulation of architecture-specific vector
(SIMD) values.

The implementation is rather loosely based on the data-parallel vector types
proposal P0214R6 for the C++ Parallelism TS 2.

## Classes

Three user-facing template classes are provided:

1. `simd<V, N, I = simd_abi::default_abi>`

   _N_-wide vector type of values of type _V_, using architecture-specific
   implementation _I_. The implementation parameter is itself a template,
   with `I<V, N>::type` being an alias for the concrete implementation
   class (see below) for _I_.

   The implementation `simd_abi::generic` provides a `std::array`-backed
   implementation for arbitrary _V_ and _N_, while `simd_abi::native`
   maps to the native architecture implementation for _V_ and _N_, if
   supported.

   `simd_abi::default_abi` will use `simd_abi::native` if available, or
   else fall back to the generic implementation.

2. `simd_mask<V, N, I = simd_api::default_abi>`

   The result of performing a lane-wise comparison/test operation on
   a `simd<V, N, I>` vector value. `simd_mask` objects support logical
   operations and are used as arguments to `where` expressions.

   `simd_mask<V, N, I>` is a type alias for `simd<V, N, I>::simd_mask`.

3. `where_expression<simd<V, N, I>>`

   The result of a `where` expression, used for masked assignment.


Implementations for specific ISAs should provide corresponding type maps
in `simd_abi`, e.g. `simd_abi::avx2<V, N>`, as well as specializing
`simd_abi::native` as appropriate.

The _scalar type_ associated with a `simd` class is the type corresponding
to one lane in the underlying representation. The _value type_ is the
type for one lane as presented in the interface. These types will typically
be the same, with the exception of `simd_mask` classes, where the
_value type_ is always `bool`.

### `simd`

Note that `simd<V, N, I>` is ultimately a type alias for `detail::simd_impl<I<V,N>::type>`,
where `I<V,N>::type` is a SIMD implementation class (see below).

In the following: `S` stands for the class `simd<V, N, I>`, and `s`
* `S` stands for the class `simd<V, N, I>`.
* `s` is an object of type `S`.
* `t`, `u` and `v` are const objects of type `S`.
* `m` is a const object of type `simd_mask<V, N, I>`.
* `i` is an index of type `int`.
* `j` is a const object of type `simd<U, N, J>` where `U` is an integral type.
* `x` is a value of type `V`.
* `p` is a pointer of type `V*`.
* `c` is a pointer of type `const V*`.

#### Type aliases and constexpr members

Name | Type | Description
-----|------|------------
`S::value_type` | `V` | The type of one lane of the SIMD type.
`S::simd_mask`  | `simd_mask<V, N, I>` | The `simd_mask` specialization resulting from `S` comparisons.
`S::width`      | `unsigned` | The SIMD width `N`.

#### Constructors

Expression | Description
-----------|------------
`S(x)`  | A SIMD value with all lanes equal to `x`.
`S(t)`  | A copy of the SIMD value `t`.
`S(c)`  | A SIMD value comprising the values `c[0]`, ..., `c[N-1]`.

#### Member functions

Expression | Type | Description
-----------|------|------------
`s.copy_from(c)`  | `void` | Set `s[0]=c[0]`, ..., `s[N-1]=c[N-1]`.
`s.gather(c, j)`  | `void` | Set `s[0]=c[j[0]]`, ..., `s[N-1]=c[j[N-1]]`.
`t.copy_to(p)`    | `void` | Set `p[0]=t[0]`, ..., `p[N-1]=t[N-1]`.
`t.scatter(p, j)` | `void` | Set `p[j[0]]=t[0]`, ..., `p[j[N-1]]=t[N-1]`.

#### Expressions

Expression | Type | Description
-----------|------|------------
`t+u` | `S`  | Lane-wise sum.
`t-u` | `S`  | Lane-wise difference.
`t*u` | `S`  | Lane-wise product.
`t/u` | `S`  | Lane-wise quotient.
`fma(t, u, v)` | `S`  | Lane-wise FMA _t_ * _u_ + _v_.
`s=t`  | `S&` | Lane-wise assignment.
`s+=t` | `S&` | Equivalent to `s=s+t`.
`s-=t` | `S&` | Equivalent to `s=s-t`.
`s*=t` | `S&` | Equivalent to `s=s*t`.
`s/=t` | `S&` | Equivalent to `s=s/t`.
`t[i]` | `V` | Value in lane _i_.
`s[i]=x` |`S::reference` | Set value in lane _i_ to `x`.
`where(s[i]=x` |`S::reference` | Set value in lane _i_ to `x`.













## Implementation requirements

A concrete implementation _C_ is expected to provide the following interface
for all

    // Architecure-specific implementation type I requires the specification of
    // the following interface, where 'a', 'b', etc. denote values of
    // `scalar_type`, and 'u', 'v', 'w', etc. denote values of `vector_type`.
    //
    // Logical operations, bool constructors and bool conversions need only be
    // provided for implementations that are used to proivde mask_type
    // operations (marked with [*] below).
    //
    // Types:
    //
    // I::vector_type                     Underlying SIMD representation type.
    // I::scalar_type                     Value type in one lane of vector_type.
    // I::mask_impl                       Implementation type for comparison results.
    // I::mask_type                       SIMD representation type for comparison results.
    //                                    (equivalent to I::mask_impl::vector_type)
    //
    // Reflection:
    //
    // constexpr static unsigned width()
    //
    // Construction:
    //
    // vector_type I::broadcast(a)        Fill SIMD type with scalar a.
    // vector_type I::broadcast(bool x)   Fill SIMD type with I::from_bool(x). [*]
    // vector_type I::immediate(a,b,...)  Populate SIMD type with given values.
    // vector_type I::immediate(bool...)  Populate SIMD type with representations of given booleans. [*]
    //
    // Load/store:
    //
    // void I::copy_to(v, scalar_type*)   Store v to memory (unaligned).
    // vector_type I::copy_from(const scalar_type*)  Load from memory (unaligned).
    //
    // Conversion:
    //
    // I::is_convertible<V>::value        True if I::convert(V) defined.
    // vector_type I::convert(V)          Convert from SIMD type V to vector_type.
    //
    // Element (lane) access:
    //
    // scalar_type I::element(u, i)       Value in ith lane of u.
    // scalar_type I::bool_element(u, i)  Boolean value in ith lane of u. [*]
    // void I::set_element(u, i, a)       Set u[i] to a.
    // void I::set_element(u, i, bool f)  Set u[i] to representation of bool f. [*]
    //
    // (Note indexing should be such that `copy_to(x, p), p[i]` should
    // have the same value as `x[i]`.)
    //
    // Arithmetic:
    //
    // vector_type I::mul(u, v)           Lane-wise multiplication.
    // vector_type I::add(u, v)           Lane-wise addition.
    // vector_type I::sub(u, v)           Lane-wise subtraction.
    // vector_type I::div(u, v)           Lane-wise division.
    // vector_type I::fma(u, v, w)        Lane-wise fused multiply-add (u*v+w).
    // (TODO: add unary minus; add bitwise operations if there is utility)
    //
    // Comparison:
    //
    // mask_type I::cmp_eq(u, v)          Lane-wise equality.
    // mask_type I::cmp_not_eq(u, v)      Lane-wise negated equality.
    // (TODO: extend)
    //
    // Logical operations:
    //
    // vector_type I::logical_not(u)      Lane-wise negation. [*]
    // vector_type I::logical_and(u, v)   Lane-wise logical and. [*]
    // vector_type I::logical_or(u, v)    Lane-wise logical or. [*]
    //
    // Mask operations:
    //
    // vector_type I::select(mask_type m, u, v)  Lane-wise m? v: u.



