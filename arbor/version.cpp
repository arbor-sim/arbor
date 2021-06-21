#include <arbor/version.hpp>

namespace arb {
const char* source_id = ARB_SOURCE_ID;
const char* arch = ARB_ARCH;
const char* build_config = ARB_BUILD_CONFIG;
const char* version = ARB_VERSION;
#ifdef ARB_VERSION_DEV
const char* version_dev = ARB_VERSION_DEV;
#else
const char* version_dev = "";
#endif
const char* full_build_id = ARB_FULL_BUILD_ID;
}
