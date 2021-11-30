#pragma once

/*
 * preprocessor macro utilities
 */

/*
 * ARB_PP_FOREACH(macro , args...)
 *   expands macro for each entry in args...
 *
 * example:
 *
 *      #define PROTO(T) T foo(T);
 *      ARB_PP_FOREACH(PROTO, int, float, double)
 *
 *  expands to
 *
 *      int foo(int); float foo(float); double foo(double);
 *
 * example:
 *
 *      #define ALLOCATE(name) int* name = new int;
 *      #define DELETE(name) delete name;
 *      #define NAMES a, b, c
 *
 *      ALLOCATE(NAMES)
 *      DELETE(NAMES)
 *
 *  expands to
 *
 *      int* a = new int; int* b = new int; int* c = new int;
 *      delete a; delete b; delete c;
*/

// Implementation macros for ARB_PP_FOREACH:

#define ARB_PP_FOREACH_1_(M, A)  M(A)
#define ARB_PP_FOREACH_2_(M, A, ...)  M(A) ARB_PP_FOREACH_1_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_3_(M, A, ...)  M(A) ARB_PP_FOREACH_2_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_4_(M, A, ...)  M(A) ARB_PP_FOREACH_3_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_5_(M, A, ...)  M(A) ARB_PP_FOREACH_4_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_6_(M, A, ...)  M(A) ARB_PP_FOREACH_5_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_7_(M, A, ...)  M(A) ARB_PP_FOREACH_6_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_8_(M, A, ...)  M(A) ARB_PP_FOREACH_7_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_9_(M, A, ...)  M(A) ARB_PP_FOREACH_8_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_10_(M, A, ...) M(A) ARB_PP_FOREACH_9_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_11_(M, A, ...) M(A) ARB_PP_FOREACH_10_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_12_(M, A, ...) M(A) ARB_PP_FOREACH_11_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_13_(M, A, ...) M(A) ARB_PP_FOREACH_12_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_14_(M, A, ...) M(A) ARB_PP_FOREACH_13_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_15_(M, A, ...) M(A) ARB_PP_FOREACH_14_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_16_(M, A, ...) M(A) ARB_PP_FOREACH_15_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_17_(M, A, ...) M(A) ARB_PP_FOREACH_16_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_18_(M, A, ...) M(A) ARB_PP_FOREACH_17_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_19_(M, A, ...) M(A) ARB_PP_FOREACH_18_(M, __VA_ARGS__)
#define ARB_PP_FOREACH_20_(M, A, ...) M(A) ARB_PP_FOREACH_19_(M, __VA_ARGS__)
#define ARB_PP_GET_21ST_ARGUMENT_(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, ...) a21

// Apply macro in first argument to each of the remaining arguments (up to 20).
// Note: if __VA_ARGS__ has size N, when it is expanded the 21st argument is the ARB_PP_FOREACH_N_ macro.
#define ARB_PP_FOREACH(M, ...)\
ARB_PP_GET_21ST_ARGUMENT_(__VA_ARGS__, ARB_PP_FOREACH_20_, ARB_PP_FOREACH_19_, ARB_PP_FOREACH_18_, ARB_PP_FOREACH_17_, ARB_PP_FOREACH_16_, ARB_PP_FOREACH_15_, ARB_PP_FOREACH_14_, ARB_PP_FOREACH_13_, ARB_PP_FOREACH_12_, ARB_PP_FOREACH_11_, ARB_PP_FOREACH_10_, ARB_PP_FOREACH_9_, ARB_PP_FOREACH_8_, ARB_PP_FOREACH_7_, ARB_PP_FOREACH_6_, ARB_PP_FOREACH_5_, ARB_PP_FOREACH_4_, ARB_PP_FOREACH_3_, ARB_PP_FOREACH_2_, ARB_PP_FOREACH_1_)(M, __VA_ARGS__)
