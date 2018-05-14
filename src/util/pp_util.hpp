#pragma once

/*
 * preprocessor macro utilities
 */

// Implementation macros for ARB_PP_FOREACH:

#define ARB_PP_FOREACH_1_(M, A, ...)  M(A)
#define ARB_PP_FOREACH_2_(M, A, ...)  M(A) ARB_PP_FOREACH_1_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_3_(M, A, ...)  M(A) ARB_PP_FOREACH_2_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_4_(M, A, ...)  M(A) ARB_PP_FOREACH_3_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_5_(M, A, ...)  M(A) ARB_PP_FOREACH_4_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_6_(M, A, ...)  M(A) ARB_PP_FOREACH_5_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_7_(M, A, ...)  M(A) ARB_PP_FOREACH_6_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_8_(M, A, ...)  M(A) ARB_PP_FOREACH_7_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_9_(M, A, ...)  M(A) ARB_PP_FOREACH_8_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_10_(M, A, ...) M(A) ARB_PP_FOREACH_9_(M, __VA_ARGS__)
#define ARB_PP_GET_11TH_ARGUMENT_(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, ...) a11

// Apply macro in first argument to each of the remaining arguments (up to 10).

#define ARB_PP_FOREACH(M, ...)\
ARB_PP_GET_11TH_ARGUMENT_(__VA_ARGS__, ARB_PP_FOREACH_10_, ARB_PP_FOREACH_9_, ARB_PP_FOREACH_8_, ARB_PP_FOREACH_7_, ARB_PP_FOREACH_6_, ARB_PP_FOREACH_5_, ARB_PP_FOREACH_4_, ARB_PP_FOREACH_3_, ARB_PP_FOREACH_2_, ARB_PP_FOREACH_1_)(M, __VA_ARGS__)
