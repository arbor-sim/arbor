Implementation of vector transcendental functions
=================================================

When building with the Intel C++ compiler, transcendental
functions on SIMD values in ``simd<double, 8, simd_detail::avx512>``
wrap calls to the Intel scalar vector mathematics library (SVML).

Outside of this case, the functions *exp*, *log*, *expm1* and
*exprelr* use explicit approximations as detailed below. The
algortihms follow those used in the
`Cephes library <http://www.netlib.org/cephes/>`_, with
some accommodations.

.. default-role:: math

Exponentials
------------

`\operatorname{exp}(x)`
^^^^^^^^^^^^^^^^^^^^^^^

The exponential is computed as

.. math::

    e^x = 2^n · e^g,

with `|g| ≤ 0.5` and `n` an integer. The power of two
is computed via direct manipulation of the exponent bits of the floating
point representation, while `e^g` is approximated by a rational polynomial.

`n` and `g` are computed by:

.. math::

    n &= \left\lfloor \frac{x}{\log 2} + 0.5 \right\rfloor

    g &= x - n·\log 2

where the subtraction in the calculation of `g` is performed in two stages,
to limit cancellation error:

.. math::

    g &\leftarrow \operatorname{fl}(x - n · c_1)

    g &\leftarrow \operatorname{fl}(g - n · c_2)

where `c_1+c_2 = \log 2`,`c_1` comprising the first 32 bits of the mantissa.
(In principle `c_1` might contain more bits of the logarithm, but this
particular decomposition matches that used in the Cephes library.) This
decomposition gives `|g|\leq \frac{1}{2}\log 2\approx 0.347`.

The rational approximation for `e^g` is of the form

.. math::

    e^g \approx \frac{R(g)}{R(-g)}

where `R(g)` is a polynomial of order 6. The coefficients are again those
used by Cephes, and probably are derived via a Remez algorithm.
`R(g)` is decomposed into even and odd terms

.. math::

    R(g) = Q(x^2) + xP(x^2)

so that the ratio can be calculated by:

.. math::

    e^g \approx 1 + \frac{2gP(g^2)}{Q(g^2)-gP(g^2)}.

Randomized testing indicates the approximation is accurate to 1 ulp.


`\operatorname{expm1}(x)`
^^^^^^^^^^^^^^^^^^^^^^^^^

A similar decomposition of `x = g + n·\log 2` is performed so that
`g≤0.5`, with the exception that `n` is always taken to
be zero for `|x|≤0.5`, i.e.

.. math::

    n = \begin{cases}
          0&\text{if $|x|≤0.5$,}\\
          \left\lfloor \frac{x}{\log 2} + 0.5 \right\rfloor
          &\text{otherwise.}
        \end{cases}


`\operatorname{expm1}(x)` is then computed as

.. math::

    e^x - 1 = 2^n·(e^g - 1)+(2^n-1).

and the same rational polynomial is used to approximate `e^g-1`,

.. math::

    e^g - 1 \approx \frac{2gP(g^2)}{Q(g^2)-gP(g^2)}.

The scaling by step for `n≠0` is in practice calculated as

.. math::

    e^x - 1 = 2·(2^{n-1}·(e^g - 1)+(2^{n-1}-0.5)).

in order to avoid overflow at the upper end of the range.

The order 6 rational polynomial approximation for small `x`
is insufficiently accurate to maintain 1 ulp accuracy; randomized
testing indicates a maximum error of up to 3 ulp.


`\operatorname{exprelr}(x)`
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function is defined as

.. math::

    \operatorname{exprelr}(x) = x/(e^x-1),

and is the reciprocal of the relative exponential function,

.. math::

    \operatorname{exprel}(x) &= {}_1F_1(1; 2; x)\\
                             &= \frac{e^x-1}{x}.

This is computed in terms of expm1 by:

.. math::

    \operatorname{exprelr}(x) :=
      \begin{cases}
          1&\text{if $\operatorname{fl}(1+x) = 1$,}\\
          x/\operatorname{expm1}(x)&\text{otherwise.}
      \end{cases}

With the approximation for `\operatorname{expm1}` used above,
randomized testing demonstrates a maximum error on the order
of 4 ulp.


Logarithms
----------

The natural logarithm is computed as

.. math::

    \log x = \log u + n·log 2

where `n` is an integer and `u` is in the interval
`[ \frac{1}{2}\sqrt 2, \sqrt 2]`. The logarithm of
`u` is then approximated by the rational polynomial
used in the Cephes implementation,

.. math::

    \log u &\approx R(u-1)

    R(z) &= z - \frac{z^2}{2} + z^3·\frac{P(z)}{Q(z)},

where `P` and `Q` are polynomials of degree 5, with
`Q` monic.

Cancellation error is minimized by computing the sum for
`\log x` as:

.. math::

    s &\leftarrow \operatorname{fl}(z^3·P(z)/Q(z))\\
    s &\leftarrow \operatorname{fl}(s + n·c_4)\\
    s &\leftarrow \operatorname{fl}(s - 0.5·z^2)\\
    s &\leftarrow \operatorname{fl}(s + z)\\
    s &\leftarrow \operatorname{fl}(s + n·c_3)

where `z=u-1` and `c_3+c_4=\log 2`, `c_3` comprising
the first 9 bits of the mantissa.

