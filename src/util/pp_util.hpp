#pragma once

/*
 * preprocessor macro utilities
 */

// Implementation macros for PP_FOREACH:

#define PP_FOREACH_1_(M, A, ...)  M(A)
#define PP_FOREACH_2_(M, A, ...)  M(A) PP_FOREACH_1_(M, __VA_ARGS__)
#define PP_FOREACH_3_(M, A, ...)  M(A) PP_FOREACH_2_(M, __VA_ARGS__)
#define PP_FOREACH_4_(M, A, ...)  M(A) PP_FOREACH_3_(M, __VA_ARGS__)
#define PP_FOREACH_5_(M, A, ...)  M(A) PP_FOREACH_4_(M, __VA_ARGS__)
#define PP_FOREACH_6_(M, A, ...)  M(A) PP_FOREACH_5_(M, __VA_ARGS__)
#define PP_FOREACH_7_(M, A, ...)  M(A) PP_FOREACH_6_(M, __VA_ARGS__)
#define PP_FOREACH_8_(M, A, ...)  M(A) PP_FOREACH_7_(M, __VA_ARGS__)
#define PP_FOREACH_9_(M, A, ...)  M(A) PP_FOREACH_8_(M, __VA_ARGS__)
#define PP_FOREACH_10_(M, A, ...) M(A) PP_FOREACH_9_(M, __VA_ARGS__)
#define PP_GET_11TH_ARGUMENT_(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, ...) a11

// Apply macro in first argument to each of the remaining arguments (up to 10).

#define PP_FOREACH(M, ...)\
PP_GET_11TH_ARGUMENT_(__VA_ARGS__, PP_FOREACH_10_, PP_FOREACH_9_, PP_FOREACH_8_, PP_FOREACH_7_, PP_FOREACH_6_, PP_FOREACH_5_, PP_FOREACH_4_, PP_FOREACH_3_, PP_FOREACH_2_, PP_FOREACH_1_)(M, __VA_ARGS__)
