SIMD classes for Arbor
======================

The purpose of the SIMD classes is to abstract and consolidate the use of
compiler intrinsics for the manipulation of architecture-specific vector
(SIMD) values.

The implementation is rather loosely based on the data-parallel vector types
proposal P0214R6 for the C++ Parallelism TS 2.

Examples
--------

The following code performs an element-wise vector product, storing
only non-zero values in the resultant array.

.. container:: example-code

    .. code-block:: cpp

        #include <util/simd.hpp>
        using namespace arb;

        void product_nonzero(int n, const double* a, const double* b, double* result) {
            constexpr int N = simd_abi::native_width<double>::value;
            using simd = simd<double, N>;
            using mask = simd::simd_mask;

            int i = 0;
            for (; i+N<=n; i+=N) {
                auto vp = simd(a+i)*simd(b+i);
                where(vp!=0, vp).copy_to(result+i);
            }

            int tail = n-i;
            auto m = mask::unpack((1<<tail)-1);

            auto vp = simd(a+i, m)*simd(b+i, m);
            where(m && vp!=0, vp).copy_to(c+i);
        }


Classes
-------

Three user-facing template classes are provided:

1. ``simd<V, N, I = simd_abi::default_abi>``

   *N*-wide vector type of values of type *V*, using architecture-specific
   implementation *I*. The implementation parameter is itself a template,
   acting as a type-map, with ``I<V, N>::type`` being the concrete implementation
   class (see below) for *N*-wide vectors of type *V* for this architecture.

   The implementation ``simd_abi::generic`` provides a ``std::array``-backed
   implementation for arbitrary *V* and *N*, while ``simd_abi::native``
   maps to the native architecture implementation for *V* and *N*, if
   supported.

   ``simd_abi::default_abi`` will use ``simd_abi::native`` if available, or
   else fall back to the generic implementation.

2. ``simd_mask<V, N, I = simd_api::default_abi>``

   The result of performing a lane-wise comparison/test operation on
   a ``simd<V, N, I>`` vector value. ``simd_mask`` objects support logical
   operations and are used as arguments to ``where`` expressions.

   ``simd_mask<V, N, I>`` is a type alias for ``simd<V, N, I>::simd_mask``.

3. ``where_expression<simd<V, N, I>>``

   The result of a ``where`` expression, used for masked assignment.

Implementation typemaps live in the ``simd_abi`` namespace, while concrete
implementation classes live in ``simd_detail``. A particular specialization
for an architecture, for example 4-wide double on AVX, then requires:

*  A concrete implementation class, e.g. ``simd_detail::avx_double4``.

*  A specialization of its ABI map, so that ``simd_abi::avx<double, 4>::type``
   is an alias for ``simd_detail::avx_double4``.

*  A specialization of the native ABI map, so that
   ``simd_abi::native<double, 4>::type`` is an alias for ``simd_abi::avx<double, 4>::type``.

The maximum natively supported width for a scalar type *V* is recorded in
``simd_abi::native_width<V>::value``.

Class ``simd``
^^^^^^^^^^^^^^

The class ``simd<V, N, I>`` is an alias for ``simd_detail::simd_impl<I<V, N>::type>``;
the class ``simd_detail::simd_impl<C>`` provides the public interface and
arithmetic operators for a concrete implementation class `C`.

In the following:

* *S* stands for the class ``simd<V, N, I>``.
* *s* is an SIMD value of type *S*.
* *m* is a mask value of type ``S::simd_mask``.
* *t*, *u* and *v* are const objects of type *S*.
* *i* is an index of type ``int``.
* *j* is a const object of type ``simd<U, N, J>`` where *U* is an integral type.
* *x* is a value of type *V*.
* *p* is a pointer to *V*.
* *c* is a const pointer to *V* or a length *N* array of *V*.

Here and below, the value in lane *i* of a SIMD vector or mask *v* is denoted by
*v*\ `i`:sub:


.. rubric:: Type aliases and constexpr members

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Name
      - Type
      - Description

    * - ``S::scalar_type``
      - *V*
      - The type of one lane of the SIMD type.

    * - ``S::simd_mask``
      - ``simd_mask<V, N, I>``
      - The ``simd_mask`` specialization resulting from comparisons of *S* SIMD values.

    * - ``S::width``
      - ``unsigned``
      - The SIMD width *N*.

.. rubric:: Constructors

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Expression
      - Description

    * - ``S(x)``
      - A SIMD value *v* with *v*\ `i`:sub: equal to *x* for *i* = 0…*N*-1.

    * - ``S(t)``
      - A copy of the SIMD value *t*.

    * - ``S(c)``
      - A SIMD value *v* with *v*\ `i`:sub: equal to ``c[i]`` for *i* = 0…*N*-1.

    * - ``S(c, m)``
      - A SIMD value *v* with *v*\ `i`:sub: equal to ``c[i]`` for *i* where *m*\ `i`:sub: is true.

.. rubric:: Member functions

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``t.copy_to(p)``
      - ``void``
      - Set ``p[i]`` to *t*\ `i`:sub: for *i* = 0…*N*-1.

    * - ``t.scatter(p, j)``
      - ``void``
      - Set ``p[j[i]]`` to *t*\ `i`:sub: for *i* = 0…*N*-1.

    * - ``s.copy_from(c)``
      - ``void``
      - Set *s*\ `i`:sub: to ``c[i]`` for *i* = 0…*N*-1.

    * - ``s.gather(c, j)``
      - ``void``
      - Set *s*\ `i`:sub: to ``c[j[i]]`` for *i* = 0…*N*-1.

.. rubric:: Expressions

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``t+u``
      - ``S``
      - Lane-wise sum.

    * - ``t-u``
      - ``S``
      - Lane-wise difference.

    * - ``t*u``
      - ``S``
      - Lane-wise product.

    * - ``t/u``
      - ``S``
      - Lane-wise quotient.

    * - ``fma(t, u, v)``
      - ``S``
      - Lane-wise FMA *t* * *u* + *v*.

    * - ``s<t``
      - ``S::simd_mask``
      - Lane-wise less-than comparison.

    * - ``s<=t``
      - ``S::simd_mask``
      - Lane-wise less-than-or-equals comparison.

    * - ``s>t``
      - ``S::simd_mask``
      - Lane-wise greater-than comparison.

    * - ``s>=t``
      - ``S::simd_mask``
      - Lane-wise greater-than-or-equals comparison.

    * - ``s==t``
      - ``S::simd_mask``
      - Lane-wise equality test.

    * - ``s!=t``
      - ``S::simd_mask``
      - Lane-wise inequality test.

    * - ``s=t``
      - ``S&``
      - Lane-wise assignment.

    * - ``s+=t``
      - ``S&``
      - Equivalent to ``s=s+t``.

    * - ``s-=t``
      - ``S&``
      - Equivalent to ``s=s-t``.

    * - ``s*=t``
      - ``S&``
      - Equivalent to ``s=s*t``.

    * - ``s/=t``
      - ``S&``
      - Equivalent to ``s=s/t``.

    * - ``s=x``
      - ``S&``
      - Equivalent to ``s=S(x)``.

    * - ``t[i]``
      - ``V``
      - Value *t*\ `i`:sub:

    * - ``s[i]=x``
      - ``S::reference``
      - Set value *s*\ `i`:sub: to *x*.

The (non-const) index operator `operator[]` returns a proxy object of type `S::reference`,
which writes the corresponding lane in the SIMD value on assignment, and has an
implicit conversion to `scalar_type`.


Class ``simd_mask``
^^^^^^^^^^^^^^^^^^^

``simd_mask<V, N, I>`` is an alias for ``simd<V, N, I>::simd_mask``, which in turn
will be an alias for a class ``simd_detail::simd_mask_impl<D>``, where *D* is
a concrete implementation class for the SIMD mask representation. ``simd_mask_impl<D>``
inherits from, and is implemented in terms of, ``simd_detail::simd_impl<D>``,
but note that the concrete implementation class *D* may or may not be the same
as the concrete implementation class ``I<V, N>::type`` used by ``simd<V, N, I>``.

Mask values are read and written as ``bool`` values of 0 or 1, which may
differ from the internal representation in each lane of the SIMD implementation.

In the following:

* *M* stands for the class ``simd_mask<V, N, I>``.
* *m* and *q* are const objects of type ``simd_mask<V, N, I>``.
* *u* is an object of type ``simd_mask<V, N, I>``.
* *b* is a boolean value.
* *w* is a pointer to ``bool``.
* *y* is a const pointer to ``bool`` or a length *N* array of ``bool``.
* *i* is of type ``int``.
* *k* is of type ``unsigned long long``.

.. rubric:: Constructors

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Expression
      - Description

    * - ``M(b)``
      - A SIMD mask *u* with *u*\ `i`:sub: equal to *b* for *i* = 0…*N*-1.

    * - ``M(m)``
      - A copy of the SIMD mask *m*.

    * - ``M(y)``
      - A SIMD value *u* with *u*\ `i`:sub: equal to ``y[i]`` for *i* = 0…*N*-1.

Note that ``simd_mask`` does not (currently) offer a masked pointer/array constructor.

.. rubric:: Member functions

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``m.copy_to(w)``
      - ``void``
      - Write the boolean value *m*\ `i`:sub: to ``w[i]`` for *i* = 0…*N*-1.

    * - ``u.copy_from(y)``
      - ``void``
      - Set *u*\ `i`:sub: to the boolean value ``y[i]`` for *i* = 0…*N*-1.

.. rubric:: Expressions

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``!m``
      - ``M``
      - Lane-wise negation.

    * - ``m&&q``
      - ``M``
      - Lane-wise logical and.

    * - ``m||q``
      - ``M``
      - Lane-wise logical or.

    * - ``m==q``
      - ``M``
      - Lane-wise equality (equivalent to ``m!=!q``).

    * - ``m!=q``
      - ``M``
      - Lane-wise logical xor.

    * - ``m=q``
      - ``M&``
      - Lane-wise assignment.

    * - ``m[i]``
      - ``bool``
      - Boolean value *m*\ `i`:sub:.

    * - ``m[i]=b``
      - ``M::reference``
      - Set *m*\ `i`:sub: to boolean value *b*.

.. rubric:: Static member functions

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``M::unpack(k)``
      - ``M``
      - Mask with value *m*\ `i`:sub: equal to the *i*\ th bit of *k*.


Class ``where_expression``
^^^^^^^^^^^^^^^^^^^^^^^^^^

``where_expression<S>`` represents a masked subset of the lanes
of a SIMD value of type ``S``, used for conditional assignment,
masked scatter, and masked gather. It is a type alias for
``S::where_expression``, and is the result of an expression of the
form ``where(mask, simdvalue)``.

In the following:

* *W* stands for the class ``where_expression<simd<V, N, I>>``.
* *s* is a reference to a SIMD value of type ``simd<V, N, I>&``.
* *t* is a SIMD value of type ``simd<V, N, I>``.
* *m* is a mask of type ``simd<V, N, I>::simd_mask``.
* *j* is a const object of type ``simd<U, N, J>`` where *U* is an integral type.
* *x* is a scalar of type *V*.
* *p* is a pointer to *V*.
* *c* is a const pointer to *V* or a length *N* array of *V*.

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``where(m, s)``
      - ``W``
      - A proxy for masked-assignment operations.

    * - ``where(m, s)=t``
      - ``void``
      - Set *s*\ `i`:sub: to *t*\ `i`:sub: for *i* where *m*\ `i`:sub: is true.

    * - ``where(m, s)=x``
      - ``void``
      - Set *s*\ `i`:sub: to *x* for *i* where *m*\ `i`:sub: is true.

    * - ``where(m, s).copy_to(p)``
      - ``void``
      - Set ``p[i]`` to *s*\ `i`:sub: for *i* where *m*\ `i`:sub: is true.

    * - ``where(m, s).scatter(p, j)``
      - ``void``
      - Set ``p[j[i]]`` to *s*\ `i`:sub: for *i* where *m*\ `i`:sub: is true.

    * - ``where(m, s).copy_from(c)``
      - ``void``
      - Set *s*\ `i`:sub: to ``c[i]`` for *i* where *m*\ `i`:sub: is true.

    * - ``where(m, s).gather(c, j)``
      - ``void``
      - Set *s*\ `i`:sub: to ``c[j[i]]`` for *i* where *m*\ `i`:sub: is true.


Top-level functions
-------------------

Lane-wise mathematical operations *abs(x)*, *min(x, y)* and *max(x, y)* are offered for
all SIMD value types, while the transcendental functions are only usable for
SIMD floating point types.

Vectorized implementations of some of the transcendental functions are provided:
refer to :doc:`simd_maths` for details.


In the following:

* *A* is a SIMD class ``simd<K, N, I>`` for some scalar type *K*.
* *S* is a SIMD class ``simd<V, N, I>`` for a floating point type *V*.
* *a* and *b* are values of type *A*.
* *s* and *t* are values of type *S*.

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``abs(a)``
      - *A*
      - Lane-wise absolute value of *a*.

    * - ``min(a, b)``
      - *A*
      - Lane-wise minimum of *a* and *b*.

    * - ``max(a, b)``
      - *A*
      - Lane-wise maximum of *a* and *b*.

    * - ``sin(s)``
      - *S*
      - Lane-wise sine of *s*.

    * - ``cos(s)``
      - *S*
      - Lane-wise cosine of *s*.

    * - ``log(s)``
      - *S*
      - Lane-wise natural logarithm of *s*.

    * - ``exp(s)``
      - *S*
      - Lane-wise exponential of *s*.

    * - ``expm1(s)``
      - *S*
      - Lane-wise :math:`x \mapsto e^x - 1`.

    * - ``exprelr(s)``
      - *S*
      - Lane-wise :math:`x \mapsto x / (e^x - 1)`.

    * - ``pow(s, t)``
      - *S*
      - Lane-wise raise *s* to the power of *t*.


Implementation requirements
---------------------------

Each specific architecture is represented by a templated class *I*, with
``I<V, N>::type`` being the concrete implementation for an *N*-wide
SIMD value with ``scalar_type`` *V*.

A concrete implementation class *C* inherits from ``simd_detail::implbase<C>``,
which provides (via CRTP) generic implementations of most of the SIMD
functionality. The base class ``implbase<C>`` in turn relies upon
``simd_detail::simd_traits<C>`` to look up the SIMD width, and associated types.

All the required SIMD operations are given by static member functions of *C*.

Minimal implementation
^^^^^^^^^^^^^^^^^^^^^^

In the following, let *C* be the concrete implementation class for a
*N*-wide vector of scalar_type *V*, with low-level representation
``archvec``.

The specialization of ``simd_detail::simd_traits<C>`` then exposes these
types and values, and also provides the concrete implementation class *M*
for masks associated with *C*:

.. container:: api-code

    .. code-block:: cpp

        template <>
        struct simd_traits<C> {
            static constexpr unsigned width = N;
            using scalar_type = V;
            using vector_type = archvec;
            using mask_impl = M;
        };


The mask implementation class *M* may or may not be the same as *C*.
For example, ``simd_detail::avx_double4`` provides both the arithmetic operations and mask
operations for an AVX 4 × double SIMD vector, while the mask
implementation for ``simd_detail::avx512_double8`` is ``simd_detail::avx512_mask8``.

The concrete implementation class must provide at minimum implementations
of ``copy_to`` and ``copy_from`` (see the section below for semantics):

.. container:: api-code

    .. code-block:: cpp

        struct C: implbase<C> {
            static void copy_to(const arch_vector&, V*);
            static arch_vector copy_from(const V*);
        };

If the implementation is also acting as a mask implementation, it must also
provide ``make_copy_to``, ``mask_copy_from``, ``mask_element`` and
``mask_set_element``:

.. container:: api-code

    .. code-block:: cpp

        struct C: implbase<C> {
            static void copy_to(const arch_vector&, V*);
            static arch_vector copy_from(const V*);

            static void mask_copy_to(const arch_vector& v, bool* w);
            static arch_vector mask_copy_from(const bool* y);
            static bool mask_element(const arch_vector& v, int i);
            static void mask_set_element(arch_vector& v, int i, bool x);
        };

The ``simd_detial::generic<T, N>`` provides an example of a minimal
implementation based on an ``arch_vector`` type of ``std::array<T, N>``.


Concrete implementation API
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following, *C* represents the concrete implementation class for
a SIMD class of width *N* and value type *V*.

* *u*, *v*, and *w* are values of type ``C::vector_type``.
* *r* is a reference of type ``C::vector_type``.
* *x* is a value of type ``C::scalar_type``.
* *c* is a const pointer of type ``const C::scalar_type*``.
* *p* is a pointer of type ``C::scalar_type*``.
* *j* is a SIMD index representation of type ``J::vector_type`` for
  an integral concrete implementation class *J*.
* *b* is a ``bool`` value.
* *w* is a pointer to ``bool``.
* *y* is a const pointer to ``bool``.
* *i* is an unsigned (index) value.
* *k* is an unsigned long long value.
* *m* is a mask representation of type ``C::mask_type``.

.. rubric:: Types and constants

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Name
      - Type
      - Description

    * - ``C::vector_type``
      - ``simd_traits<C>::vector_type``
      - Underlying SIMD representation type.

    * - ``C::scalar_type``
      - ``simd_traits<C>::scalar_type``
      - Should be convertible to/from *V*.

    * - ``C::mask_impl``
      - ``simd_traits<C>::mask_impl``
      - Concrete implementation class for mask SIMD type.

    * - ``C::mask_type``
      - ``C::mask_impl::vector_type``
      - Underlying SIMD representation for masks.

    * - ``C::width``
      - ``unsigned``
      - The SIMD width *N*.

.. rubric:: Initialization, load, store

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``C::broadcast(x)``
      - ``C::vector_type``
      - Fill representation with scalar *x*.

    * - ``C::copy_to(v, p)``
      - ``void``
      - Store values *v*\ `i`:sub: to *p+i*. *p* may be unaligned.

    * - ``C::copy_to_masked(v, p, m)``
      - ``void``
      - Store values *v*\ `i`:sub: to *p+i* wherever *m*\ `i`:sub: is true. *p* may be unaligned.

    * - ``C::copy_from(c)``
      - ``C::vector_type``
      - Return a vector with values *v*\ `i`:sub: loaded from *p+i*. *p* may be unaligned.

    * - ``C::copy_from_masked(c, m)``
      - ``C::vector_type``
      - Return a vector with values *v*\ `i`:sub: loaded from *p+i* wherever *m*\ `i`:sub: is true. *p* may be unaligned.

    * - ``C::copy_from_masked(w, c, m)``
      - ``void``
      - Return a vector with values *v*\ `i`:sub: loaded from *p+i* wherever *m*\ `i`:sub: is true, or equal to *w*\ `i`:sub
        otherwise. *p* may be unaligned.

.. rubric:: Lane access

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``C::element(v, i)``
      - ``C::scalar_type``
      - Value in ith lane of *u*.

    * - ``C::set_element(r, i, x)``
      - ``void``
      - Set value in lane *i* of *r* to *x*.

.. rubric:: Gather and scatter

The offsets for gather and scatter operations are given
by a vector type ``J::vector_type`` for some possibly
different concrete implementation class *J*, and the
static methods implementing gather and scatter are templated
on this class.

Implementations can provide optimized versions for specific
index classes *J*; this process would be simplified with
more support for casts between SIMD types and their concrete
implementations, functionality which is not yet provided.

The first argument to these functions is a dummy argument
of type *J*, used only to disambiguate overloads.

.. list-table::
    :header-rows: 1
    :widths: 20 20 60

    * - Expression
      - Type
      - Description

    * - ``C::gather(J{}, p, j)``
      - ``C::vector_type``
      - Vector *v* with values *v*\ `i`:sub: = ``p[j[i]]``.

    * - ``C::gather(J{}, u, p, j, m)``
      - ``C::vector_type``
      - Vector *v* with values *v*\ `i`:sub: = *m*\ `i`:sub: ? ``p[j[i]]`` : *u*\ `i`:sub:.

    * - ``C::scatter(J{}, u, p, j)``
      - ``void``
      - Write values *u*\ `i`:sub: to ``p[j[i]]``.

    * - ``C::scatter(J{}, u, p, j, m)``
      - ``void``
      - Write values *u*\ `i`:sub: to ``p[j[i]]`` for lanes *i* where *m*\ `i`:sub: is true.

.. rubric:: Arithmetic operations

.. list-table::
    :header-rows: 1
    :widths: 20 20 60

    * - Expression
      - Type
      - Description

    * - ``C::negate(v)``
      - ``C::vector_type``
      - Lane-wise unary minus.

    * - ``C::mul(u, v)``
      - ``C::vector_type``
      - Lane-wise multiplication.

    * - ``C::add(u, v)``
      - ``C::vector_type``
      - Lane-wise addition.

    * - ``C::sub(u, v)``
      - ``C::vector_type``
      - Lane-wise subtraction.

    * - ``C::div(u, v)``
      - ``C::vector_type``
      - Lane-wise division.

    * - ``C::fma(u, v, w)``
      - ``C::vector_type``
      - Lane-wise fused multiply-add (u*v+w).

.. rubric:: Comparison and blends

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``C::cmp_eq(u, v)``
      - ``C::mask_type``
      - Lane-wise *u* = *v*.

    * - ``C::cmp_neq(u, v)``
      - ``C::mask_type``
      - Lane-wise *u* ≠ *v*.

    * - ``C::cmp_gt(u, v)``
      - ``C::mask_type``
      - Lane-wise *u* > *v*.

    * - ``C::cmp_geq(u, v)``
      - ``C::mask_type``
      - Lane-wise *u* ≥ *v*.

    * - ``C::cmp_lt(u, v)``
      - ``C::mask_type``
      - Lane-wise *u* < *v*.

    * - ``C::cmp_leq(u, v)``
      - ``C::mask_type``
      - Lane-wise *u* ≤ *v*.

    * - ``C::ifelse(m, u, v)``
      - ``C::vector_type``
      - Vector *w* with values *w*\ `i`:sub: = *m*\ `i`:sub: ? *u*\ `i`:sub: : *v*\ `i`:sub:.

.. rubric:: Mathematical function support.

With the exception of ``abs``, ``min`` and ``max``, these are only
required for floating point vector implementations.

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``C::abs(v)``
      - ``C::vector_type``
      - Lane-wise absolute value.

    * - ``C::min(u, v)``
      - ``C::vector_type``
      - Lane-wise minimum.

    * - ``C::max(u, v)``
      - ``C::vector_type``
      - Lane-wise maximum.

    * - ``C::sin(v)``
      - ``C::vector_type``
      - Lane-wise sine.

    * - ``C::cos(v)``
      - ``C::vector_type``
      - Lane-wise cosine.

    * - ``C::log(v)``
      - ``C::vector_type``
      - Lane-wise natural logarithm.

    * - ``C::exp(v)``
      - ``C::vector_type``
      - Lane-wise exponential.

    * - ``C::expm1(v)``
      - ``C::vector_type``
      - Lane-wise :math:`x \mapsto e^x -1`.

    * - ``C::exprelr(v)``
      - ``C::vector_type``
      - Lane-wise :math:`x \mapsto x/(e^x -1)`.

    * - ``C::pow(u, v)``
      - ``C::vector_type``
      - Lane-wise *u* raised to the power of *v*.

.. rubric:: Mask value support

Mask operations are only required if *C* constitutes the implementation of a
SIMD mask class.

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``C::mask_broadcast(b)``
      - ``C::vector_type``
      - Fill mask representation with bool *b*.

    * - ``C::mask_element(v, i)``
      - ``bool``
      - Mask value *v*\ `i`:sub:.

    * - ``C::mask_set_element(u, i, b)``
      - ``void``
      - Set mask value *u*\ `i`:sub: to *b*.

    * - ``C::mask_copy_to(v, w)``
      - ``void``
      - Write bool values to memory (unaligned).

    * - ``C::mask_copy_from(y)``
      - ``C::vector_type``
      - Load bool values from memory (unaligned).

    * - ``C::mask_unpack(k)``
      - ``C::vector_type``
      - Return vector *v* with boolean value *v*\ `i`:sub: equal
        to the *i*\ th bit of *k*.

.. rubric:: Logical operations

Logical operations are only required if *C* constitutes the implementation of a
SIMD mask class.

.. list-table::
    :header-rows: 1
    :widths: 20 20 60

    * - Expression
      - Type
      - Description

    * - ``C::logical_not(u)``
      - ``C::vector_type``
      - Lane-wise negation.

    * - ``C::logical_and(u, v)``
      - ``C::vector_type``
      - Lane-wise logical and.

    * - ``C::logical_or(u, v)``
      - ``C::vector_type``
      - Lane-wise logical or.

    * - ``C::select(m, v, w)``
      - ``C::vector_type``
      - Lane-wise *m*? *v*: *u*.


Missing functionality
---------------------

There is no support yet for the following features, although some of these
will need to be provided in order to improve the efficiency of SIMD versions
of our generated mechanisms.

* A SIMD cast function, e.g. ``simd_cast<S>(const T&)`` that converts between
  different SIMD wrappers of the same width. The infrastructure that supports
  this in concrete implementation classes would also simplify the implementation
  of more generic ``gather`` and ``scatter`` methods.

* Horizontal reductions across the lanes of a SIMD value or where-expression.

* Vectorizable implementations of trigonometric functions.

* Compound assignment operations for where-expressions. Extending the concrete
  implementation API to support this would allow, for example, efficient use
  of AVX512 masked arithmetic instructions.

