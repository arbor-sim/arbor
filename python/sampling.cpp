#include <memory>

#include <model.hpp>

#include "sampling.hpp"

spike_recorder make_spike_recorder(arb::model& m) {
    spike_recorder r;
    m.set_global_spike_callback(r.callback());
    return r;
}
