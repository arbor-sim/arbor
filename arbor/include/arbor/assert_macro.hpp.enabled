#pragma once

#ifdef __GNUC__
    #define ARB_DEBUG_FUNCTION_NAME_ __PRETTY_FUNCTION__
#else
    #define ARB_DEBUG_FUNCTION_NAME_ __func__
#endif

#define arb_assert(condition) \
(void)((condition) || \
(arb::global_failed_assertion_handler(#condition, __FILE__, __LINE__, ARB_DEBUG_FUNCTION_NAME_), true))
