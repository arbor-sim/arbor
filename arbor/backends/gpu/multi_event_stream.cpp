#include <arbor/common_types.hpp>

#include "backends/gpu/multi_event_stream.hpp"
#include "memory/memory.hpp"

namespace arb {
namespace gpu {

void multi_event_stream_base::clear() {
    memory::fill(span_begin_, 0u);
    memory::fill(span_end_, 0u);
    memory::fill(mark_, 0u);
    n_nonempty_stream_[0] = 0;
}

} // namespace gpu
} // namespace arb
