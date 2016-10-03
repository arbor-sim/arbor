set(LIB_SOURCES
    fill.cu
)

if( CUDA_BACKEND )
    cuda_add_library(gpu ${LIB_SOURCES})
endif()

