
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

    // it might be acceptable to have the entire builder defined here
    // because the storage might need to be back end specific
    struct matrix_assembler {
        view d;
        view u;
        view rhs;
        const_iview p;

        const_view sigma;
        const_view alpha;
        const_view voltage;
        const_view current;
        const_view cv_capacitance;

        array alpha_d;

        matrix_assembler() = default;

        matrix_assembler(
            view d, view u, view rhs, const_iview p,
            const_view sigma, const_view alpha,
            const_view voltage, const_view current, const_view cv_capacitance)
        :
            d{d}, u{u}, rhs{rhs}, p{p},
            sigma{sigma}, alpha{alpha},
            voltage{voltage}, current{current}, cv_capacitance{cv_capacitance}
        {
            auto n = d.size();
            alpha_d = array(n, 0);
            for(auto i: util::make_span(1u, n)) {
                alpha_d[i] += alpha[i];

                // add contribution to the diagonal of parent
                alpha_d[p[i]] += alpha[i];
            }
        }

        void build(value_type dt) {
            auto n = d.size();
            value_type factor_lhs = 1e5*dt;
            value_type factor_rhs = 1e1*dt; //  units: 10·ms/(F/m^2)·(mA/cm^2) ≡ mV
            for (auto i: util::make_span(0u, n)) {
                d[i] = sigma[i] + factor_lhs*alpha_d[i];
                u[i] = -factor_lhs*alpha[i];
                // the RHS of the linear system is
                //      cv_area * (V - dt/cm*(im - ie))
                rhs[i] = sigma[i]*(voltage[i] - factor_rhs/cv_capacitance[i]*current[i]);
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
        const std::vector<size_type>& node_indices)
    {
        if (!has_mechanism(name)) {
            throw std::out_of_range("no mechanism in database : " + name);
        }

        return mech_map_.find(name)->second(vec_v, vec_i, iarray(node_indices));
    }

    static bool has_mechanism(const std::string& name) { return mech_map_.count(name)>0; }

    static std::string name() {
        return "multicore";
    }

private:

    using maker_type = mechanism (*)(view, view, iarray&&);
    static std::map<std::string, maker_type> mech_map_;

    template <template <typename> class Mech>
    static mechanism maker(view vec_v, view vec_i, iarray&& node_indices) {
        return mechanisms::make_mechanism<Mech<backend>>
            (vec_v, vec_i, std::move(node_indices));
    }
};

} // namespace multicore
} // namespace mc
} // namespace nest
