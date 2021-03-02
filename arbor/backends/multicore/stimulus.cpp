#include <cmath>

#include <arbor/fvm_types.hpp>
#include <arbor/mechanism_ppack.hpp>

#include "backends/builtin_mech_proto.hpp"
#include "backends/multicore/mechanism.hpp"


namespace arb {
namespace multicore {

struct stimulus_pp: ::arb::mechanism_ppack {
    fvm_value_type* delay;
    fvm_value_type* duration;
    fvm_value_type* amplitude;
};

class stimulus: public arb::multicore::mechanism {
public:
    const mechanism_fingerprint& fingerprint() const override {
        static mechanism_fingerprint hash = "##builtin_stimulus";
        return hash;
    }
    std::string internal_name() const override { return "_builtin_stimulus"; }
    mechanismKind kind() const override { return ::arb::mechanismKind::point; }
    mechanism_ptr clone() const override { return mechanism_ptr(new stimulus()); }

    void init() override {}
    void advance_state() override {}
    void compute_currents() override {
        size_type n = size();
        for (size_type i=0; i<n; ++i) {
            auto cv = pp_.node_index_[i];
            auto t = pp_.vec_t_[pp_.vec_di_[cv]];

            if (t>=pp_.delay[i] && t<pp_.delay[i]+pp_.duration[i]) {
                // Amplitudes are given as a current into a compartment, so subtract.
                pp_.vec_i_[cv] -= pp_.weight_[i]*pp_.amplitude[i];
            }
        }
    }
    void write_ions() override {}
    void apply_events(deliverable_event_stream::state events) override {}

protected:
    std::size_t object_sizeof() const override { return sizeof(*this); }
    virtual mechanism_ppack* ppack_ptr() override { return &pp_; }

    mechanism_field_table field_table() override {
        return {
            {"delay", &pp_.delay},
            {"duration", &pp_.duration},
            {"amplitude", &pp_.amplitude}
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
    stimulus_pp pp_;
};
} // namespace multicore

template <>
concrete_mech_ptr<multicore::backend> make_builtin_stimulus() {
    return concrete_mech_ptr<multicore::backend>(new arb::multicore::stimulus());
}

} // namespace arb
