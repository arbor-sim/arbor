#include <cmath>

#include <backends/builtin_mech_proto.hpp>
#include <backends/multicore/fvm.hpp>
#include <util/indirect.hpp>

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
        auto vec_t = util::indirect_view(util::indirect_view(vec_t_, vec_ci_), node_index_);
        auto vec_i = util::indirect_view(vec_i_, node_index_);
        size_type n = size();
        for (size_type i=0; i<n; ++i) {
            auto t = vec_t[i];
            if (t>=delay[i] && t<delay[i]+duration[i]) {
                // Amplitudes are given as a current into a compartment, so subtract.
                vec_i[i] -= weight_[i]*amplitude[i];
            }
        }
    }
    void write_ions() override {}
    void deliver_events(deliverable_event_stream::state events) override {}

protected:
    std::size_t object_sizeof() const override { return sizeof(*this); }

    mechanism_field_table field_table() {
        return mechanism_field_table{
            {"delay", &delay, 0},
            {"duration", &duration, 0},
            {"amplitude", &amplitude, 0}
        };
    }

private:
    view delay;
    view duration;
    view amplitude;
};
} // namespace multicore

template <>
concrete_mech_ptr<multicore::backend> make_builtin_stimulus() {
    return concrete_mech_ptr<multicore::backend>(new arb::multicore::stimulus());
}

} // namespace arb
