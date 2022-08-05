#ifndef ARB_TYPES_H
#define ARB_TYPES_H

// Define ABI arb_ typedefs.

#ifdef __cplusplus
#include <cstdint>
using std::uint32_t;
#else
#include <stdint.h>
#endif

#ifdef __cplusplus
namespace arb {
#endif

#include <arbor/arb_types.inc>

#ifdef __cplusplus
}
#endif

#endif // ARB_TYPES_H
