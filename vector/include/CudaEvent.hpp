#pragma once

#include <limits>
#include <memory>

#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Event.hpp"

namespace memory {

// wrapper around cuda events
class CudaEvent
: public AsynchEvent {
public:

    //////////////////////////////////
    // constructor
    //////////////////////////////////
    CudaEvent() {
        //cudaError_t status = cudaEventCreate(&event_);
        //assert(status == cudaSuccess);
        cudaEvent_t tmp;
        cudaError_t status = cudaEventCreate(&tmp);
        event_ = std::shared_ptr<cudaEvent_t>(new cudaEvent_t(tmp));
#ifdef VERBOSE
        std::cout << "CudaEvent< " << event() << " > :: create" << std::endl;
#endif
        assert(status == cudaSuccess);
    }

    ////////////////////////////////////////////////////////////////////
    // destructor
    ////////////////////////////////////////////////////////////////////
    // there is no need to wait for event to finish:
    // in the case that an event has been recorded and not yet completed,
    // cudaEventDestroy() will return immediately, and the resources associated
    // with the event will be released automatically when the event finishes.
    // Furthermore once an cudaEvent_t has been used to synchronize a stream,
    // it can be destroyed and the stream won't be affected (it will still
    // synchronize on the event)
    ~CudaEvent() {
#ifdef VERBOSE
        std::cout << "CudaEvent< " << event() << " > :: destruct count = "
                  << event_.use_count()
                  << " " << (event_.unique() ? " deleting" : "")
                  << std::endl;
#endif
        if(event_.unique())
            cudaEventDestroy(event());
    }

    CudaEvent(CudaEvent const& other) : event_(other.event_) {}

    ////////////////////////////////////////////////////////////////////
    // return an event handle
    ////////////////////////////////////////////////////////////////////
    cudaEvent_t& event() {
        return *event_;
    }

    ////////////////////////////////////////////////////////////////////
    // force host execution to wait for event completion
    ////////////////////////////////////////////////////////////////////
    virtual void wait() override {
        cudaError_t status = cudaEventSynchronize(event());
        assert(status == cudaSuccess);
    }

    virtual EventStatus query() override {
        cudaError_t status = cudaEventQuery(event());

        // assert that expected results have been returned
        assert(status==cudaSuccess || status==cudaErrorNotReady);

        // cudaSuccess means that task finished, or that cudaEventRecord() was
        // not called on event_
        if(status==cudaErrorNotReady)
            return kEventBusy;
        return kEventReady;
    }

    ////////////////////////////////////////////////////////////////////
    // returns time in seconds taken between this cuda event and another
    // cuda event
    ////////////////////////////////////////////////////////////////////
    // returns NaN if there is an error
    // time is this - other
    double time_since(CudaEvent &other) {
        float time_taken = 0.0f;

        cudaError_t status =
            cudaEventElapsedTime(&time_taken, other.event(), event());
        if(status != cudaSuccess)
            return std::numeric_limits<double>::quiet_NaN();
        return double(time_taken/1.e3);
    }

private:

    std::shared_ptr<cudaEvent_t> event_;
};

} // namespace memory

    /*CudaEvent(CudaEvent&& other) {
        event_ = std::move(other.event_);
    }*/
