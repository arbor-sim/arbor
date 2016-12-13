
#include <map>
#include <string>

#include <common_types.hpp>
#include <mechanism.hpp>
#include <memory/memory.hpp>
#include <util/span.hpp>

namespace nest {
namespace mc {
namespace multicore {

struct backend {
    /// define the real and index types
    using value_type = double;
    using size_type  = nest::mc::cell_lid_type;

    /// define storage types
    using array  = memory::host_vector<value_type>;
    using iarray = memory::host_vector<size_type>;

    using view       = typename array::view_type;
    using const_view = typename array::const_view_type;

    using iview       = typename iarray::view_type;
    using const_iview = typename iarray::const_view_type;

    using host_array  = array;
    using host_iarray = iarray;

    using host_view   = view;
    using host_iview  = iview;

    static void hines_solve(
        view d, view u, view rhs,
        const_iview p, const_iview cell_index)
    {
        const size_type ncells = cell_index.size()-1;

        // loop over submatrices
        for (auto m: util::make_span(0, ncells)) {
            auto first = cell_index[m];
            auto last = cell_index[m+1];

            // backward sweep
            for(auto i=last-1; i>first; --i) {
                auto factor = u[i] / d[i];
                d[p[i]]   -= factor * u[i];
                rhs[p[i]] -= factor * rhs[i];
            }
            rhs[first] /= d[first];

            // forward sweep
            for(auto i=first+1; i<last; ++i) {
                rhs[i] -= u[i] * rhs[p[i]];
                rhs[i] /= d[i];
            }
        }
    }

    struct matrix_assembler {
        view d;     // [μS]
        view u;     // [μS]
        view rhs;   // [nA]
        const_iview p;

        const_view cv_capacitance;      // [pF]
        const_view face_conductance;    // [μS]
        const_view voltage;             // [mV]
        const_view current;             // [nA]

        // the invariant part of the matrix diagonal
        array invariant_d;              // [μS]

        matrix_assembler() = default;

        matrix_assembler(
            view d, view u, view rhs, const_iview p,
            const_view cv_capacitance,
            const_view face_conductance,
            const_view voltage,
            const_view current)
        :
            d{d}, u{u}, rhs{rhs}, p{p},
            cv_capacitance{cv_capacitance}, face_conductance{face_conductance},
            voltage{voltage}, current{current}
        {
            auto n = d.size();
            invariant_d = array(n, 0);
            for (auto i: util::make_span(1u, n)) {
                auto gij = face_conductance[i];

                u[i] = -gij;
                invariant_d[i] += gij;
                invariant_d[p[i]] += gij;
            }
        }

        void assemble(value_type dt) {
            auto n = d.size();
            value_type factor = 1e-3/dt;
            for (auto i: util::make_span(0u, n)) {
                auto gi = factor*cv_capacitance[i];

                d[i] = gi + invariant_d[i];

                rhs[i] = gi*voltage[i] - current[i];
            }
        }
    };

    //
    // mechanism infrastructure
    //
    using ion = mechanisms::ion<backend>;

    using mechanism = mechanisms::mechanism_ptr<backend>;

    static mechanism make_mechanism(
        const std::string& name,
        view vec_v, view vec_i,
        const std::vector<value_type>& weights,
        const std::vector<size_type>& node_indices)
    {
        if (!has_mechanism(name)) {
            throw std::out_of_range("no mechanism in database : " + name);
        }

        return mech_map_.find(name)->second(vec_v, vec_i, array(weights), iarray(node_indices));
    }

    static bool has_mechanism(const std::string& name) {
        return mech_map_.count(name)>0;
    }

    static std::string name() {
        return "cpu";
    }

private:

    using maker_type = mechanism (*)(view, view, array&&, iarray&&);
    static std::map<std::string, maker_type> mech_map_;

    template <template <typename> class Mech>
    static mechanism maker(view vec_v, view vec_i, array&& weights, iarray&& node_indices) {
        return mechanisms::make_mechanism<Mech<backend>>
            (vec_v, vec_i, std::move(weights), std::move(node_indices));
    }
};

} // namespace multicore
} // namespace mc
} // namespace nest

#include "stimulus_multicore.hpp"

