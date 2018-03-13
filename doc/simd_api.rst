SIMD classes for Arbor
======================

The purpose of the SIMD classes is to abstract and consolidate the use of
compiler intrinsics for the manipulation of architecture-specific vector
(SIMD) values.

The implementation is rather loosely based on the data-parallel vector types
proposal P0214R6 for the C++ Parallelism TS 2.

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


Class ``simd``
^^^^^^^^^^^^^^

The class ``simd<V, N, I>`` is an alias for ``simd_detail::simd_impl<I<V, N>::type>``;
the class ``simd_detail::simd_impl<C>`` provides the public interface and
arithmetic operators for a concrete implementation class `C`.

In the following:

* ``S`` stands for the class ``simd<V, N, I>``.
* ``s`` is an SIMD value of type ``S``.
* ``m`` is a mask value of type ``S::simd_mask``.
* ``t``, ``u`` and ``v`` are const objects of type ``S``.
* ``i`` is an index of type ``int``.
* ``j`` is a const object of type ``simd<U, N, J>`` where ``U`` is an integral type.
* ``x`` is a value of type ``V``.
* ``p`` is a pointer of type ``V*``.
* ``c`` is a pointer of type ``const V*`` or a value of type ``V[N]`` (taken by reference).


.. rubric:: Type aliases and constexpr members

.. list-table:: 
    :widths: 20 20 60
    :header-rows: 1

    * - Name
      - Type
      - Description

    * - ``S::scalar_type``
      - ``V``
      - The type of one lane of the SIMD type.

    * - ``S::simd_mask``
      - ``simd_mask<V, N, I>``
      - The ``simd_mask`` specialization resulting from ``S`` comparisons.

    * - ``S::width``
      - ``unsigned``
      - The SIMD width ``N``.

.. rubric:: Constructors

.. list-table:: 
    :widths: 20 80
    :header-rows: 1

    * - Expression
      - Description

    * - ``S(x)``
      - A SIMD value with all lanes equal to ``x``.

    * - ``S(t)``
      - A copy of the SIMD value ``t``.

    * - ``S(c)``
      - A SIMD value comprising the values ``c[0]``, ..., ``c[N-1]``.

.. rubric:: Member functions

.. list-table:: 
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``t.copy_to(p)``
      - ``void``
      - Set ``p[0]=t[0]``, ..., ``p[N-1]=t[N-1]``.

    * - ``t.scatter(p, j)``
      - ``void``
      - Set ``p[j[0]]=t[0]``, ..., ``p[j[N-1]]=t[N-1]``.

    * - ``t.scatter(p, j, m)``
      - ``void``
      - Set ``p[j[i]]=t[i]`` where ``m[i]`` is true.

    * - ``s.copy_from(c)``
      - ``void``
      - Set ``s[0]=c[0]``, ..., ``s[N-1]=c[N-1]``.

    * - ``s.gather(c, j)``
      - ``void``
      - Set ``s[0]=c[j[0]]``, ..., ``s[N-1]=c[j[N-1]]``.

    * - ``s.gather(c, j, m)``
      - ``void``
      - Set ``s[i]=c[j[i]]`` where ``m[i]`` is true.

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
      - Value in lane *i*.

    * - ``s[i]=x``
      - ``S::reference``
      - Set value in lane *i* to ``x``.

The (non-const) index operator `operator[]` returns a proxy object of type `S::reference`,
which writes the corresponding lane in the SIMD value on assignment, and has an
implicit conversion to `scalar_type`.


Class ``simd_mask``
^^^^^^^^^^^^^^^^^^^

``simd_mask<V, N, I>`` is an alias for ``simd<V, N, I>::simd_mask``, which in turn
will be an alias for a class ``simd_detail::simd_mask_impl<D>``, where ``D`` is
a concrete implementation class for the SIMD mask representation. ``simd_mask_impl<D>``
inherits from and is implemented in terms of ``simd_detail::simd_impl<D>``,
but note that the concrete implementation class ``D`` may or may not be the same
as the concrete implementation class ``I<V, N>::type`` used by ``simd<V, N, I>``.

Mask values are read and written as ``bool`` values of 0 or 1, which may
differ from the internal representation in each lane of the SIMD implementation.

In the following:

* ``M`` stands for the class ``simd_mask<V, N, I>``.
* ``m`` and ``q`` are const objects of type ``simd_mask<V, N, I>``.
* ``u`` is an object of type ``simd_mask<V, N, I>``.
* ``b`` is a boolean value.
* ``w`` is a pointer of type ``bool*``.
* ``y`` is a pointer of type ``const bool*``.

.. rubric:: Constructors

.. list-table:: 
    :widths: 20 80
    :header-rows: 1

    * - Expression
      - Description

    * - ``M(b)``
      - A SIMD mask with all lanes equal to ``b``.

    * - ``M(q)``
      - A copy of the SIMD mask ``q``.

    * - ``M(y)``
      - A SIMD value comprising the values ``v[0]``, ..., ``v[N-1]``.

.. rubric:: Member functions

.. list-table:: 
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``m.copy_to(w)``
      - ``void``
      - Set ``m[0]=w[0]``, ..., ``m[N-1]=w[N-1]``.

    * - ``u.copy_from(y)``
      - ``void``
      - Set ``y[0]=u[0]``, ..., ``y[N-1]=u[N-1]``.

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
      - Lane-wise and.

    * - ``m||q``
      - ``M``
      - Lane-wise or.

    * - ``m==q``
      - ``M``
      - Lane-wise equality (equivalent to ``m!=!q``).

    * - ``m!=q``
      - ``M``
      - Lane-wise xor.

    * - ``m=q``
      - ``M&``
      - Lane-wise assignment.

    * - ``m[i]``
      - ``bool``
      - Value in lane *i*.

    * - ``m[i]=b``
      - ``M::reference``
      - Set value in lane *i* to ``b``.


Class ``where_expression``
^^^^^^^^^^^^^^^^^^^^^^^^^^

``where_expression<S>`` represents a masked subset of the lanes
of a SIMD value of type ``S``, used for conditional assignment,
masked scatter, and masked gather. It is a type alias for
``S::where_expression``, and is the result of an expression of the
form ``where(mask, simdvalue)``.


In the following:

* ``W`` stands for the class ``where_expression<simd<V, N, I>>``.
* ``s`` is a reference to a SIMD value of type ``simd<V, N, I>&``.
* ``t`` is a SIMD value of type ``simd<V, N, I>``.
* ``x`` is a scalar of type *V*.
* ``m`` is a mask of type ``simd<V, N, I>::simd_mask``.
* ``j`` is a const object of type ``simd<U, N, J>`` where ``U`` is an integral type.
* ``p`` is a pointer of type ``V*``.
* ``c`` is a pointer of type ``const V*`` or a value of type ``V[N]`` (taken by reference).

.. list-table:: 
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``where(m, s)``
      - ``W``
      - A proxy for masked-assignment operation.

    * - ``where(m, s)=t``
      - ``void``
      - Set ``s[i]=t[i]`` for *i* such that ``m[i]``.

    * - ``where(m, s)=x``
      - ``void``
      - Set ``s[i]=x`` for *i* such that ``m[i]``.

    * - ``where(m, s).gather(c, j)``
      - ``void``
      - Set ``s[i]=c[j[i]]`` for *i* such that ``m[i]``.

    * - ``where(m, s).scatter(p, j)``
      - ``void``
      - Set ``p[j[i]]=c[i]`` for *i* such that ``m[i]``.

Example
-------

The following code performs an element-wise vector product, storing
only non-zero values1
stroing

Implementation requirements
---------------------------

Each specific architecture is represented by a templated class ``I``, with
``I<V, N>::type`` being the concrete implementation for an *N*-wide
SIMD value with ``value_type`` *V*. Any such concrete implementation
``C`` must provide the following interface to support the SIMD value
operations.

``simd_mask`` types are also implemented in terms of a concrete implementation
class. Operations that are required only for ``simd_mask`` functionality
are marked with (*) below, and are not otherwise required.

In the following, ``C`` represents the concrete implementation class for
a SIMD class of width ``N`` and value type ``V``.

* ``v`` and ``w`` are values of type ``C::vector_type``.
* ``u`` is a reference of type ``C::vector_type``.
* ``x`` is a value of type ``C::scalar_type``.
* ``p`` is a pointer of type ``C::scalar_type*``.
* ``b`` is a bool value.
* ``w`` is a pointer to bool.
* ``y`` is a const pointer to bool.
* ``i`` is an unsigned (index) value.
* ``m`` is a mask representation of type ``C::mask_type``.

.. rubric:: Types

.. list-table:: 
    :widths: 20 20 60
    :header-rows: 1

    * - Name
      - Type
      - Description

    * - ``C::vector_type``
      - *implementation defined*
      - Underlying SIMD representation type.

    * - ``C::scalar_type``
      - *implementation defined*
      - Should be convertible to/from *V*.

    * - ``C::mask_impl``
      - *implementation defined*
      - Concrete implementation class for mask SIMD type.

    * - ``C::mask_type``
      - ``C::mask_type::vector_type``
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
      - Fill representation with scalar *a*.

    * - ``C::copy_to(v, p)``
      - ``void``
      - Store v to memory (unaligned).

    * - ``C::copy_from(p)``
      - ``C::vector_type``
      - Load from memory (unaligned).

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

    * - ``C::set_element(u, i, x)``
      - ``void``
      - Set value in lane *i* of *u* to *x*.

.. rubric:: Mask value support

.. list-table:: 
    :widths: 20 20 60
    :header-rows: 1

    * - ``C::mask_broadcast(b)``
      - ``C::vector_type``
      - Fill mask representation with bool *b*. (*)

    * - ``C::mask_element(v, i)``
      - ``bool``
      - Mask value in ith lane of *v*. (*)

    * - ``C::mask_set_element(u, i, b)``
      - ``void``
      - Set mask value in lane *i* of *u* to *b*. (*)

    * - ``C::mask_copy_to(v, w)``
      - ``void``
      - Write bool values to memory (unaligned). (*)

    * - ``C::mask_copy_from(y)``
      - ``C::vector_type``
      - Load bool values from memory (unaligned). (*)

.. rubric:: Arithmetic and logical operations

.. list-table:: 
    :header-rows: 1
    :widths: 20 20 60

    * - Expression
      - Type
      - Description

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

    * - ``C::logical_not(u)``
      - ``C::vector_type``
      - Lane-wise negation. (*)

    * - ``C::logical_and(u, v)``
      - ``C::vector_type``
      - Lane-wise logical and. (*)

    * - ``C::logical_or(u, v)``
      - ``C::vector_type``
      - Lane-wise logical or. (*)

    * - ``C::select(m, v, w)``
      - ``C::vector_type``
      - Lane-wise *m* ? *v*: *u*.

.. rubric:: Comparison

.. list-table:: 
    :widths: 20 20 60
    :header-rows: 1

    * - Expression
      - Type
      - Description

    * - ``C::cmp_eq(v, w)``
      - ``C::mask_type``
      - Lane-wise *v* = *w*.

    * - ``C::cmp_neq(v, w)``
      - ``C::mask_type``
      - Lane-wise *v* ≠ *w*.

    * - ``C::cmp_gt(v, w)``
      - ``C::mask_type``
      - Lane-wise *v* > *w*.

    * - ``C::cmp_geq(v, w)``
      - ``C::mask_type``
      - Lane-wise *v* ≥ *w*.

    * - ``C::cmp_lt(v, w)``
      - ``C::mask_type``
      - Lane-wise *v* &lt; *w*.

    * - ``C::cmp_leq(v, w)``
      - ``C::mask_type``
      - Lane-wise *v* ≤ *w*.


Gather/scatter
^^^^^^^^^^^^^^

Gather/scatter operations require in addition to the participating
SIMD value to read or write, a SIMD value of indices to describe
the offsets. Default implementations are provided by templated
classes in ``simd_detail``:

* ``simd_detail::gather_impl<Impl, ImplIndex>``
* ``simd_detail::masked_gather_impl<Impl, ImplIndex>``
* ``simd_detail::scatter_impl<Impl, ImplIndex>``
* ``simd_detail::masked_scatter_gather_impl<Impl, ImplIndex>``

Here, ``Impl`` represents the concerete implementation class for
the SIMD value, and ``ImplIndex`` the concrete implementation class
for the SIMD index.

The default implementations copy the SIMD data to standard C-style
arrays and perform the loads and stores explicitly.
Architecture-specific optimizations are then provided by specializing
these implementation classes.

Specializing gather operations
""""""""""""""""""""""""""""""

Unmasked gather is provided by the static method ::

    vector_type gather_impl<Impl, ImplIndex>::gather(const scalar_type* p, const index_type& index)

where ``vector_type`` is ``Impl::vector_type``, the raw representation of the SIMD data,
``scalar_type`` is ``Impl::scalar_type``, the per-lane type for the SIMD data, and ``index_type``
is ``ImplIndex::vector_type``, the raw representation of the SIMD index.

The method returns a raw SIMD value with lane values given by ``p[index[i]]`` for each lane ``i``.

An implementation for a specific architecture specializes the template and implements this
static method. For example, the ``AVX2`` gather implementation for 4-wide ``double`` values
and ``int`` offsets (within the ``simd_detail`` namespace)::

    template <typename Impl, typename ImplIndex>
    struct gather_impl;

    template <>
    struct gather_impl<avx2_double4, avx2_int4> {
        static __m256d gather(const double* p, const __m128i& index) {
            return  _mm256_i32gather_pd(p, index, 8);
        };
    };

This provides an intrinsics-based implementation for the method
``simd<double, 4, simd_avi::avx2>::gather(const double*, const simd<int, 4, simd_avi::avx2>)``

Masked gather is provided by ::

    vector_type masked_gather_impl<Impl, ImplIndex>::gather(
        vector_type a, const scalar_type* p, const index_type& index, const mask_type& mask)

where ``mask_type`` is the raw SIMD representation for the mask associated with Impl, i.e.
``Impl::mask_impl::vector_type``.

The method returns a raw SIMD value with lane values given by ``mask[i]? p[index[i]]: a[i]``.

Architectural specialization is performed similarly.

#### Specializing scatter operations

TBC

### Casting

TBC

