#include <vector>
#include <numeric>

#include <arbor/common_types.hpp>
#include <arbor/spike.hpp>

#include "threading/enumerable_thread_specific.hpp"
#include "threading/threading.hpp"
#include "thread_private_spike_store.hpp"

namespace arb {

struct local_spike_store_type {
    threading::enumerable_thread_specific<std::vector<spike>> buffers_;

    local_spike_store_type(const task_system_handle& ts): buffers_(ts) {};
};

thread_private_spike_store::thread_private_spike_store(thread_private_spike_store&& t):
    impl_(std::move(t.impl_))
{}

thread_private_spike_store::thread_private_spike_store(const task_system_handle& ts):
    impl_(new local_spike_store_type(ts))
{}

thread_private_spike_store::~thread_private_spike_store() = default;

std::vector<spike> thread_private_spike_store::gather() const {
    const auto& bs = impl_->buffers_;
    auto len = std::accumulate(bs.begin(), bs.end(), 0u, [](auto acc, const auto& b) { return acc + b.size(); });
    std::vector<spike> spikes;
    spikes.reserve(len);
    std::for_each(bs.begin(), bs.end(), [&spikes] (const auto& b) { spikes.insert(spikes.end(), b.begin(), b.end()); });
    return spikes;
}

std::vector<spike>& thread_private_spike_store::get() {
    return impl_->buffers_.local();
}

void thread_private_spike_store::clear() {
    for (auto& b: impl_->buffers_) {
        b.clear();
    }
}
} // namespace arb
