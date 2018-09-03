#include <pybind11/pybind11.h>

#include <arbor/common_types.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/schedule.hpp>

namespace arb {
namespace py {

struct event_generator {
    arb::cell_lid_type lid;
    double weight;
    arb::schedule time_seq;

    event_generator(arb::cell_lid_type lid, double weight, arb::schedule seq):
        lid(lid),
        weight(weight),
        time_seq(std::move(seq))
    {}
};

} // namespace arb
} // namespace py

