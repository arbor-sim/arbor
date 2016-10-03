/*
 * Driver program for the unit tests
 */

#include "gtest.h"

//#include <mpi.h>

#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

bool initialize_cuda() {
    CUresult result = cuInit(0);
    return result == CUDA_SUCCESS;
}
#endif

int main(int argc, char **argv) {
    //MPI_Init(&argc, &argv);

    #ifdef WITH_CUDA
    // if testing with CUDA turned on, first check that we can initialize CUDA
    if(!initialize_cuda()) {
        std::cerr << "ERROR: unable to initialize CUDA" << std::endl;
        return 1;
    }
    #endif

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    //MPI_Finalize();
}

