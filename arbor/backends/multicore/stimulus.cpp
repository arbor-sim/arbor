#include <cmath>

#include <arbor/fvm_types.hpp>

#include "backends/builtin_mech_proto.hpp"
#include "backends/multicore/mechanism.hpp"

namespace arb {

namespace multicore {
class stimulus: public arb::multicore::mechanism {
public:
    const mechanism_fingerprint& fingerprint() const override {
        static mechanism_fingerprint hash = "##builtin_stimulus";
        return hash;
    }
    std::string internal_name() const override { return "_builtin_stimulus"; }
    mechanismKind kind() const override { return ::arb::mechanismKind::point; }
    mechanism_ptr clone() const override { return mechanism_ptr(new stimulus()); }

    void nrn_init() override {}
    void nrn_state() override {}
    void nrn_current() override {
        size_type n = size();
        for (size_type i=0; i<n; ++i) {
            auto cv = node_index_[i];
            auto t = vec_t_[vec_ci_[cv]];

            if (t>=delay[i] && t<delay[i]+duration[i]) {
                // Amplitudes are given as a current into a compartment, so subtract.
                vec_i_[cv] -= weight_[i]*amplitude[i];
            }
        }
    }
    void write_ions() override {}
    void deliver_events(deliverable_event_stream::state events) override {}

protected:
    std::size_t object_sizeof() const override { return sizeof(*this); }

    mechanism_field_table field_table() override {
        return {
            {"delay", &delay},
            {"duration", &duration},
            {"amplitude", &amplitude}
        };
    }

    mechanism_field_default_table field_default_table() override {
        return {
            {"delay", 0},
            {"duration", 0},
            {"amplitude", 0}
        };
    }
private:
    fvm_value_type* delay;
    fvm_value_type* duration;
    fvm_value_type* amplitude;
};
} // namespace multicore

template <>
concrete_mech_ptr<multicore::backend> make_builtin_stimulus() {
    return concrete_mech_ptr<multicore::backend>(new arb::multicore::stimulus());
}

} // namespace arb
