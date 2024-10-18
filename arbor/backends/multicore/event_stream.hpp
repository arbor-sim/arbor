#pragma once

// Indexed collection of pop-only event queues --- multicore back-end implementation.

#include "backends/event_stream_base.hpp"

namespace arb {
namespace multicore {

template<typename BaseEventStream>
struct event_stream : BaseEventStream {
  public:
    ARB_SERDES_ENABLE(event_stream<BaseEventStream>,
                      ev_data_,
                      ev_spans_,
                      index_);
  protected:
    void init() override final {
        this->base_ptr_ = this->ev_data_.data();
    }
};

using spike_event_stream = event_stream<spike_event_stream_base>;
using sample_event_stream = event_stream<sample_event_stream_base>;

} // namespace multicore
} // namespace arb
