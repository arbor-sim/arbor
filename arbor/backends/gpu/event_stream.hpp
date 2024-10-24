#pragma once

// Indexed collection of pop-only event queues --- CUDA back-end implementation.

#include "backends/event_stream_base.hpp"
#include "memory/memory.hpp"

namespace arb {
namespace gpu {

template<typename BaseEventStream>
struct event_stream : BaseEventStream {
  public:
    ARB_SERDES_ENABLE(event_stream<BaseEventStream>,
                      ev_data_,
                      ev_spans_,
                      device_ev_data_,
                      index_);

  protected:
    void init() override final {
        resize(this->device_ev_data_, this->ev_data_.size());
        memory::copy_async(this->ev_data_, this->device_ev_data_);
        this->base_ptr_ = this->device_ev_data_.data();
    }

  private: // device memory
    using event_data_type = typename BaseEventStream::event_data_type;
    using device_array = memory::device_vector<event_data_type>;

    device_array device_ev_data_;

    template<typename D>
    static void resize(D& d, std::size_t size) {
        // resize if necessary
        if (d.size() < size) {
            d = D(size);
        }
    }
};

using spike_event_stream = event_stream<spike_event_stream_base>;
using sample_event_stream = event_stream<sample_event_stream_base>;

} // namespace gpu
} // namespace arb
