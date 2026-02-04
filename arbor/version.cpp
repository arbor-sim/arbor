#include <arbor/version.hpp>
#include <arbor/export.hpp>
#include <string>

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

/* somehow cannot be found in arb namespace... (error during examples linking)
ARB_ARBOR_API std::string get_arbor_config_str() {
    std::string config_str = "";
    #ifdef ARB_MPI_ENABLED
        config_str += std::string("mpi=true, ");
    #else
        config_str += std::string("mpi=false, ");
    #endif
    #ifdef ARB_NVCC_ENABLED
        config_str += std::string("cuda=true, ");
    #endif
    #ifdef ARB_CUDA_CLANG_ENABLED
        config_str += std::string("cuda-clang=true, ");
    #endif
    #ifdef ARB_HIP_ENABLED
        config_str += std::string("hip=true, ");
    #endif
    #ifndef ARB_GPU_ENABLED
        config_str += std::string("gpu=false, ");
    #endif
    #ifdef ARB_VECTORIZE_ENABLED
        config_str += std::string("vectorize=true, ");
    #else
        config_str += std::string("vectorize=false, ");
    #endif
    #ifdef ARB_PROFILE_ENABLED
        config_str += std::string("profiling=true, ");
    #else
        config_str += std::string("profiling=false, ");
    #endif
    #ifdef ARB_NEUROML_ENABLED
        config_str += std::string("neuroml=true, ");
    #else
        config_str += std::string("neuroml=false, ");
    #endif
    #ifdef ARB_BUNDLED_ENABLED
        config_str += std::string("bundled=true, ");
    #else
        config_str += std::string("bundled=false");
    #endif
    config_str += std::string("version='") + ARB_VERSION + "', " +
                  std::string("source='") + ARB_SOURCE_ID + "', " +
                  std::string("build_config='") + ARB_BUILD_CONFIG + "', " +
                  std::string("arch='") + ARB_ARCH + "'";
    return config_str;
}*/
}
