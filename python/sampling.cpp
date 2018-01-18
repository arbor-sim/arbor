#include <memory>

#include <model.hpp>

#include "sampling.hpp"

std::shared_ptr<spike_recorder> make_spike_recorder(arb::model& m) {
    auto r = std::make_shared<spike_recorder>();
    m.set_global_spike_callback(r->callback());
    return r;
}
