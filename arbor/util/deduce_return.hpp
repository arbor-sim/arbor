#pragma once

/* Just one macro, to fill the gap before C++14 */

#ifdef DEDUCED_RETURN_TYPE
#undef DEDUCED_RETURN_TYPE
#endif

#define DEDUCED_RETURN_TYPE(expr) -> decltype(expr) { return expr; }

