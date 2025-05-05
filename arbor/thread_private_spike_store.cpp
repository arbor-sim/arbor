#include <vector>
#include <numeric>

#include <arbor/common_types.hpp>
#include <arbor/spike.hpp>

#include "threading/enumerable_thread_specific.hpp"
#include "threading/threading.hpp"
#include "thread_private_spike_store.hpp"

namespace arb {

struct local_spike_store_type {
    threading::enumerable_thread_specific<std::vector<std::vector<spike>>> buffers_;

    local_spike_store_type(const task_system_handle& ts): buffers_(ts) {};
};

thread_private_spike_store::thread_private_spike_store(thread_private_spike_store&& t):
    impl_(std::move(t.impl_))
{}

thread_private_spike_store::thread_private_spike_store(const task_system_handle& ts):
    impl_(new local_spike_store_type(ts))
{}

thread_private_spike_store::~thread_private_spike_store() = default;

std::vector<std::vector<spike>> thread_private_spike_store::gather() const {
    const auto& bs = impl_->buffers_;
    std::vector<std::vector<spike>> result;
    result.reserve(bs.size());
    for (const auto& thread_buffers : bs) {
        result.insert(result.end(), thread_buffers.begin(), thread_buffers.end());
    }
    return result;
}

std::vector<std::vector<spike>>& thread_private_spike_store::get() {
    return impl_->buffers_.local();
}

void thread_private_spike_store::clear() {
    for (auto& b: impl_->buffers_) {
        b.clear();
    }
}
} // namespace arb
