#include <arbor/mechanism.hpp>

#include "profile/profiler_macro.hpp"

namespace arb {
void mechanism::initialize() {
    PROFILE_ZONE();
    ANNOTATE_ZONE(mech_.name, strlen(mech_.name));
    iface_.init_mechanism(&ppack_);
}

void mechanism::update_current() {
    PROFILE_ZONE();
    ANNOTATE_ZONE(mech_.name, strlen(mech_.name));
    iface_.compute_currents(&ppack_);
}

void mechanism::update_state() {
    PROFILE_ZONE();
    ANNOTATE_ZONE(mech_.name, strlen(mech_.name));
    iface_.advance_state(&ppack_);
}

void mechanism::update_ions() {
    PROFILE_ZONE();
    ANNOTATE_ZONE(mech_.name, strlen(mech_.name));
    iface_.write_ions(&ppack_);
}

void mechanism::post_event() {
    PROFILE_ZONE();
    ANNOTATE_ZONE(mech_.name, strlen(mech_.name));
    iface_.post_event(&ppack_);
}

void mechanism::deliver_events(arb_deliverable_event_stream& stream) {
    PROFILE_ZONE();
    ANNOTATE_ZONE(mech_.name, strlen(mech_.name));
    iface_.apply_events(&ppack_, &stream);
}
}
