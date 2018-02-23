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

In the following:
* `S` stands for the class `simd<V, N, I>`.
* `s` is an object of type `S`.
* `t`, `u` and `v` are const objects of type `S`.
* `i` is an index of type `int`.
* `j` is a const object of type `simd<U, N, J>` where `U` is an integral type.
* `x` is a value of type `V`.
* `p` is a pointer of type `V*`.
* `c` is a pointer of type `const V*` or a value of type `V[N]` (taken by reference).

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
`t.copy_to(p)`    | `void` | Set `p[0]=t[0]`, ..., `p[N-1]=t[N-1]`.
`t.scatter(p, j)` | `void` | Set `p[j[0]]=t[0]`, ..., `p[j[N-1]]=t[N-1]`.
`s.copy_from(c)`  | `void` | Set `s[0]=c[0]`, ..., `s[N-1]=c[N-1]`.
`s.gather(c, j)`  | `void` | Set `s[0]=c[j[0]]`, ..., `s[N-1]=c[j[N-1]]`.

#### Expressions

Expression | Type | Description
-----------|------|------------
`t+u`  | `S` | Lane-wise sum.
`t-u`  | `S` | Lane-wise difference.
`t*u`  | `S` | Lane-wise product.
`t/u`  | `S` | Lane-wise quotient.
`fma(t, u, v)` | `S` | Lane-wise FMA _t_ * _u_ + _v_.
`s<t`  | `S::simd_mask` | Lane-wise less-than comparison.
`s<=t` | `S::simd_mask` | Lane-wise less-than-or-equals comparison.
`s>t`  | `S::simd_mask` | Lane-wise greater-than comparison.
`s>=t` | `S::simd_mask` | Lane-wise greater-than-or-equals comparison.
`s==t` | `S::simd_mask` | Lane-wise equality test.
`s!=t` | `S::simd_mask` | Lane-wise inequality test.
`s=t`  | `S&` | Lane-wise assignment.
`s+=t` | `S&` | Equivalent to `s=s+t`.
`s-=t` | `S&` | Equivalent to `s=s-t`.
`s*=t` | `S&` | Equivalent to `s=s*t`.
`s/=t` | `S&` | Equivalent to `s=s/t`.
`s=x`  | `S&` | Equivalent to `s=S(x)`.
`t[i]` | `V` | Value in lane _i_.
`s[i]=x` |`S::reference` | Set value in lane _i_ to `x`.


### `simd_mask`

In the following:
* `M` stands for the class `simd_mask<V, N, I>`.
* `m` and `q` are const objects of type `simd_mask<V, N, I>`.
* `u` is an object of type `simd_mask<V, N, I>`.
* `b` is a boolean value.
* `w` is a pointer of type `bool*`.
* `y` is a pointer of type `const bool*`.

#### Constructors

Expression | Description
-----------|------------
`M(b)`  | A SIMD mask with all lanes equal to `b`.
`M(q)`  | A copy of the SIMD mask `q`.
`M(y)`  | A SIMD value comprising the values `v[0]`, ..., `v[N-1]`.

#### Member functions

Expression | Type | Description
-----------|------|------------
`m.copy_to(w)`    | `void` | Set `m[0]=w[0]`, ..., `m[N-1]=w[N-1]`.
`u.copy_from(y)`  | `void` | Set `y[0]=u[0]`, ..., `y[N-1]=u[N-1]`.

#### Expressions

Expression | Type | Description
-----------|------|------------
`!m`   | `M` | Lane-wise negation.
`m&&q` | `M` | Lane-wise and.
`m||q` | `M` | Lane-wise or.
`m==q` | `M` | Lane-wise equality (equivalent to `m!=!q`).
`m!=q` | `M` | Lane-wise xor.
`m=q`  | `M&` | Lane-wise assignment.
`m[i]` | `bool` | Value in lane _i_.
`m[i]=b` |`M::reference` | Set value in lane _i_ to `b`.


### `where_expression`

In the following:
* `W` stands for the class `where_expression<simd<V, N, I>>`.
* `s` is a reference to a SIMD value of type `simd<V, N, I>&`.
* `t` is a SIMD value of type `simd<V, N, I>`.
* `x` is a scalar of type _V_.
* `m` is a mask of type `simd<V, N, I>::simd_mask`.

Expression | Type | Description
-----------|------|------------
`where(m, s)` | W | A proxy for masked-assignment operation.
`where(m, s)=t` |`void` | Set `s[i]=t[i]` for _i_ such that `m[i]`.
`where(m, s)=x` |`void` | Set `s[i]=x` for _i_ such that `m[i]`.

## Casting

The `simd_cast<S>(T)` top-level function converts a SIMD-value of
type `T` to one of type `S`, provided they are of both the same
length, and the value types of `T` is explicitly convertible to
the value type of `S`.

## Implementation requirements

Each specific architecture is represented by a templated class `I`, with
`I<V, N>::type` being the concrete implementation for an _N_-wide
SIMD value with `value_type` _V_. Any such concrete implementation
`C` must provide the following interface to support the SIMD value
operations.

`simd_mask` types are also implemented in terms of a concrete implementation
class. Operations that are required only for `simd_mask` functionality
are marked with (*) below, and are not otherwise required.

In the following, `C` represents the concrete implementation class for
a SIMD class of width `N` and value type `V`.
* `v` and `w` are values of type `C::vector_type`.
* `u` is a reference of type `C::vector_type`.
* `x` is a value of type `C::scalar_type`.
* `p` is a pointer of type `C::scalar_type*`.
* `b` is a bool value.
* `w` is a pointer to bool.
* `y` is a const pointer to bool.
* `i` is an unsigned (index) value.
* `m` is a mask representation of type `C::mask_type`.

#### Types

Name | Type | Description
-----|------|------------
`C::vector_type` | _implementation defined_ | Underlying SIMD representation type.
`C::scalar_type` | _implementation defined_ | Should be convertible to/from _V_.
`C::mask_impl`   | _implementation defined_ | Concrete implementation class for mask SIMD type.
`C::mask_type`   | `C::mask_type::vector_type` | Underlying SIMD representation for masks.
`C::width`       | `unsigned` | The SIMD width _N_.

#### Initialization, load, store

Expression | Type | Description
-----------|------|------------
`C::broadcast(x)`  | `C::vector_type` | Fill representation with scalar _a_.
`C::copy_to(v, p)` | `void` | Store v to memory (unaligned).
`C::copy_from(p)`  | `C::vector_type` | Load from memory (unaligned).

#### Lane access

Expression | Type | Description
-----------|------|------------
`C::element(v, i)` | `C::scalar_type` | Value in ith lane of _u_.
`C::set_element(u, i, x)` | `void` | Set value in lane _i_ of _u_ to _x_.

#### Mask value support

`C::mask_broadcast(b)`  | `C::vector_type` | Fill mask representation with bool _b_. (*)
`C::mask_element(v, i)` | `bool`           | Mask value in ith lane of _v_. (*)
`C::mask_set_element(u, i, b)` | `void`    | Set mask value in lane _i_ of _u_ to _b_. (*)
`C::mask_copy_to(v, w)` | `void`           | Write bool values to memory (unaligned). (*)
`C::mask_copy_from(y)`  | `C::vector_type` | Load bool values from memory (unaligned). (*)


#### Arithmetic and logical operations

Expression | Type | Description
-----------|------|------------
`C::mul(u, v)`       | `C::vector_type` |  Lane-wise multiplication.
`C::add(u, v)`       | `C::vector_type` |  Lane-wise addition.
`C::sub(u, v)`       | `C::vector_type` |  Lane-wise subtraction.
`C::div(u, v)`       | `C::vector_type` |  Lane-wise division.
`C::fma(u, v, w)`    | `C::vector_type` |  Lane-wise fused multiply-add (u*v+w).
`C::logical_not(u)`    | `C::vector_type` |  Lane-wise negation. (*)
`C::logical_and(u, v)` | `C::vector_type` |  Lane-wise logical and. (*)
`C::logical_or(u, v)`  | `C::vector_type` |  Lane-wise logical or. (*)
`C::select(m, v, w)` | `C::vector_type` |  Lane-wise _m_ ? _v_: _u_.

#### Comparison

Expression | Type | Description
-----------|------|------------
`C::cmp_eq(v, w)`  | `C::mask_type` | Lane-wise _v_ = _w_.
`C::cmp_neq(v, w)` | `C::mask_type` | Lane-wise _v_ ≠ _w_.
`C::cmp_gt(v, w)`  | `C::mask_type` | Lane-wise _v_ > _w_.
`C::cmp_geq(v, w)` | `C::mask_type` | Lane-wise _v_ ≥ _w_.
`C::cmp_lt(v, w)`  | `C::mask_type` | Lane-wise _v_ &lt; _w_.
`C::cmp_leq(v, w)` | `C::mask_type` | Lane-wise _v_ ≤ _w_.

### Gather/scatter

Gather/scatter operations require in addition to the participating
SIMD value to read or write, a SIMD value of indices to describe
the offsets. Default implementations are provided by templated
classes in `simd_detail`:

* `simd_detail::gather_impl<Impl, ImplIndex>`
* `simd_detail::masked_gather_impl<Impl, ImplIndex>`
* `simd_detail::scatter_impl<Impl, ImplIndex>`
* `simd_detail::masked_scatter_gather_impl<Impl, ImplIndex>`

Here, `Impl` represents the concerete implementation class for
the SIMD value, and `ImplIndex` the concrete implementation class
for the SIMD index.

The default implementations copy the SIMD data to standard C-style
arrays and perform the loads and stores explicitly.
Architecture-specific optimizations are then provided by specializing
these implementation classes.

#### Specializing gather operations

Unmasked gather is provided by the static method
```
vector_type gather_impl<Impl, ImplIndex>::gather(
    const scalar_type* p, const index_type& index)`
```
where `vector_type` is `Impl::vector_type`, the raw representation of the SIMD data,
`scalar_type` is `Impl::scalar_type`, the per-lane type for the SIMD data, and `index_type`
is `ImplIndex::vector_type`, the raw representation of the SIMD index.

The method returns a raw SIMD value with lane values given by `p[index[i]]` for each lane `i`.

An implementation for a specific architecture specializes the template and implements this
static method. For example, the `AVX2` gather implementation for 4-wide `double` values
and `int` offsets (within the `simd_detail` namespace):
```
template <typename Impl, typename ImplIndex>
struct gather_impl;

template <>
struct gather_impl<avx2_double4, avx2_int4> {
    static __m256d gather(const double* p, const __m128i& index) {
        return  _mm256_i32gather_pd(p, index, 8);
    };
};

```
This provides an intrinsics-based implementation for the method
`simd<double, 4, simd_avi::avx2>::gather(const double*, const simd<int, 4, simd_avi::avx2>)`

Masked gather is provided by
```
vector_type masked_gather_impl<Impl, ImplIndex>::gather(
    vector_type a, const scalar_type* p, const index_type& index, const mask_type& mask)
```
where `mask_type` is the raw SIMD representation for the mask associated with Impl, i.e.
`Impl::mask_impl::vector_type`.

The method returns a raw SIMD value with lane values given by `mask[i]? p[index[i]]: a[i]`.

Architectural specialization is performed similarly.

#### Specializing scatter operations

TBC

### Casting

TBC

