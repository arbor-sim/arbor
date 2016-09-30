#pragma once

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaEvent.hpp"

namespace memory {

// wrapper around cuda streams
class CudaStream {
public:
    ////////////////////////////////////////////////////////////////////
    // default constructor
    // sets to the default stream 0
    ////////////////////////////////////////////////////////////////////
    CudaStream() : stream_(0) {};

    ////////////////////////////////////////////////////////////////////
    // constructor with flag for whether or not to create a new stream
    ////////////////////////////////////////////////////////////////////
    // if no stream is to be created, then default stream 0 is used
    CudaStream(bool create_new_stream) {
        stream_ = 0;
        if(create_new_stream)
            stream_ = new_stream();
    }

    ////////////////////////////////////////////////////////////////////
    // destructor
    ////////////////////////////////////////////////////////////////////
    ~CudaStream() {
        if(stream_) {
            cudaError_t status = cudaStreamDestroy(stream_);
            assert(status == cudaSuccess);
        }
    }

    ////////////////////////////////////////////////////////////////////
    // returns boolean indicating whether this is the default (NULL) stream
    ////////////////////////////////////////////////////////////////////
    bool is_default_stream() {
        return stream_==0;
    }

    ////////////////////////////////////////////////////////////////////
    // return the cuda Stream handle
    ////////////////////////////////////////////////////////////////////
    cudaStream_t stream() {
        return stream_;
    }

    ////////////////////////////////////////////////////////////////////
    // insert event into stream
    ////////////////////////////////////////////////////////////////////
    // returns immediately
    CudaEvent insert_event() {
        CudaEvent e;

        cudaError_t status = cudaEventRecord(e.event(), stream_);
        assert(status == cudaSuccess);

        return e;
    }

    ////////////////////////////////////////////////////////////////////
    // make all future work on stream wait until event has completed.
    ////////////////////////////////////////////////////////////////////
    // returns immediately, not waiting for event to complete
    void wait_on_event(CudaEvent &e) {
        cudaError_t status = cudaStreamWaitEvent(stream_, e.event(), 0);

        assert(status == cudaSuccess);
    }

private:

    ////////////////////////////////////////////////////////////////////
    // helper that creates a new CUDA stream using CUDA API
    ////////////////////////////////////////////////////////////////////
    cudaStream_t new_stream() {
        cudaStream_t s;
        cudaError_t status = cudaStreamCreate(&s);

        assert(status == cudaSuccess);

        return s;
    }

    cudaStream_t stream_;
};

} // namespace memory
