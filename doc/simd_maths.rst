Implementation of vector transcendental functions
=================================================

When building with the Intel C++ compiler, transcendental
functions on SIMD values in ``simd<double, 8, simd_detail::avx512>``
wrap calls to the Intel scalar vector mathematics library (SVML).

Outside of this case, the functions *exp*, *log*, *expm1* and
*exprelr* use explicit approximations as detailed below.

Exponentials
------------

TBC

Logarithms
----------

TBC

