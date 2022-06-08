#include <arbor/version.hpp>
#include <arbor/export.hpp>

namespace arb {
ARB_ARBOR_API const char* source_id = ARB_SOURCE_ID;
ARB_ARBOR_API const char* arch = ARB_ARCH;
ARB_ARBOR_API const char* build_config = ARB_BUILD_CONFIG;
ARB_ARBOR_API const char* version = ARB_VERSION;
#ifdef ARB_VERSION_DEV
ARB_ARBOR_API const char* version_dev = ARB_VERSION_DEV;
#else
ARB_ARBOR_API const char* version_dev = "";
#endif
ARB_ARBOR_API const char* full_build_id = ARB_FULL_BUILD_ID;
}
