#include "backends/builtin_mech_proto.hpp"
#include "backends/gpu/mechanism.hpp"
#include "backends/gpu/mechanism_ppack_base.hpp"

#include "stimulus.hpp"

namespace arb {
namespace gpu {

class stimulus: public arb::gpu::mechanism {
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
        stimulus_current_impl(size(), pp_);
    }

    void write_ions() override {}
    void deliver_events(deliverable_event_stream::state events) override {}

    mechanism_ppack_base* ppack_ptr() override {
        return &pp_;
    }

protected:
    std::size_t object_sizeof() const override { return sizeof(*this); }

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

} // namespace gpu

template <>
concrete_mech_ptr<gpu::backend> make_builtin_stimulus() {
    return concrete_mech_ptr<gpu::backend>(new arb::gpu::stimulus());
}

} // namespace arb
