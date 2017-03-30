
#include <map>
#include <string>

#include <common_types.hpp>
#include <mechanism.hpp>
#include <memory/memory.hpp>
#include <memory/wrappers.hpp>
#include <util/span.hpp>

#include "stimulus_multicore.hpp"

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

    /// matrix state
    struct matrix_state {
        iarray parent_index;
        iarray cell_index;

        array d;     // [μS]
        array u;     // [μS]
        array rhs;   // [nA]

        array cv_capacitance;      // [pF]
        array face_conductance;    // [μS]

        // the invariant part of the matrix diagonal
        array invariant_d;         // [μS]

        const_view solution;

        matrix_state() = default;

        matrix_state( const std::vector<size_type>& p,
                      const std::vector<size_type>& cell_idx,
                      const std::vector<value_type>& cap,
                      const std::vector<value_type>& cond):
            parent_index(memory::make_const_view(p)),
            cell_index(memory::make_const_view(cell_idx)),
            d(size(), 0), u(size(), 0), rhs(size()),
            cv_capacitance(memory::make_const_view(cap)),
            face_conductance(memory::make_const_view(cond))
        {
            EXPECTS(cap.size() == size());
            EXPECTS(cond.size() == size());
            EXPECTS(cell_idx.back() == size());

            auto n = d.size();
            invariant_d = array(n, 0);
            for (auto i: util::make_span(1u, n)) {
                auto gij = face_conductance[i];

                u[i] = -gij;
                invariant_d[i] += gij;
                invariant_d[p[i]] += gij;
            }

            // In this back end the solution is a simple view of the rhs, which
            // contains the solution after the matrix_solve is performed.
            solution = rhs;
        }

        std::size_t size() const {
            return parent_index.size();
        }

        // Assemble the matrix
        // Afterwards the diagonal and RHS will have been set given dt, voltage and current
        //   dt      [ms]
        //   voltage [mV]
        //   current [nA]
        void assemble(value_type dt, const_view voltage, const_view current) {
            auto n = d.size();
            value_type factor = 1e-3/dt;
            for (auto i: util::make_span(0u, n)) {
                auto gi = factor*cv_capacitance[i];

                d[i] = gi + invariant_d[i];

                rhs[i] = gi*voltage[i] - current[i];
            }
        }

        void solve() {
            const size_type ncells = cell_index.size()-1;

            // loop over submatrices
            for (auto m: util::make_span(0, ncells)) {
                auto first = cell_index[m];
                auto last = cell_index[m+1];

                // backward sweep
                for(auto i=last-1; i>first; --i) {
                    auto factor = u[i] / d[i];
                    d[parent_index[i]]   -= factor * u[i];
                    rhs[parent_index[i]] -= factor * rhs[i];
                }
                rhs[first] /= d[first];

                // forward sweep
                for(auto i=first+1; i<last; ++i) {
                    rhs[i] -= u[i] * rhs[parent_index[i]];
                    rhs[i] /= d[i];
                }
            }
        }
    };

    //
    // mechanism infrastructure
    //
    using ion = mechanisms::ion<backend>;

    using mechanism = mechanisms::mechanism_ptr<backend>;

    using stimulus = mechanisms::multicore::stimulus<backend>;

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

    /// threshold crossing logic
    /// used as part of spike detection back end
    class threshold_watcher {
    public:
        /// stores a single crossing event
        struct threshold_crossing {
            size_type index;    // index of variable
            value_type time;    // time of crossing
            friend bool operator== (
                const threshold_crossing& lhs, const threshold_crossing& rhs)
            {
                return lhs.index==rhs.index && lhs.time==rhs.time;
            }
        };

        threshold_watcher() = default;

        threshold_watcher(
                const_view vals,
                const std::vector<size_type>& indxs,
                const std::vector<value_type>& thresh,
                value_type t=0):
            values_(vals),
            index_(memory::make_const_view(indxs)),
            thresholds_(memory::make_const_view(thresh)),
            v_prev_(vals)
        {
            is_crossed_ = iarray(size());
            reset(t);
        }

        /// Remove all stored crossings that were detected in previous calls
        /// to the test() member function.
        void clear_crossings() {
            crossings_.clear();
        }

        /// Reset state machine for each detector.
        /// Assume that the values in values_ have been set correctly before
        /// calling, because the values are used to determine the initial state
        void reset(value_type t=0) {
            clear_crossings();
            for (auto i=0u; i<size(); ++i) {
                is_crossed_[i] = values_[index_[i]]>=thresholds_[i];
            }
            t_prev_ = t;
        }

        const std::vector<threshold_crossing>& crossings() const {
            return crossings_;
        }

        /// The time at which the last test was performed
        value_type last_test_time() const {
            return t_prev_;
        }

        /// Tests each target for changed threshold state
        /// Crossing events are recorded for each threshold that
        /// is crossed since the last call to test
        void test(value_type t) {
            for (auto i=0u; i<size(); ++i) {
                auto v_prev = v_prev_[i];
                auto v      = values_[index_[i]];
                auto thresh = thresholds_[i];
                if (!is_crossed_[i]) {
                    if (v>=thresh) {
                        // the threshold has been passed, so estimate the time using
                        // linear interpolation
                        auto pos = (thresh - v_prev)/(v - v_prev);
                        auto crossing_time = t_prev_ + pos*(t - t_prev_);
                        crossings_.push_back({i, crossing_time});

                        is_crossed_[i] = true;
                    }
                }
                else {
                    if (v<thresh) {
                        is_crossed_[i] = false;
                    }
                }

                v_prev_[i] = v;
            }
            t_prev_ = t;
        }

        bool is_crossed(size_type i) const {
            return is_crossed_[i];
        }

        /// the number of threashold values that are being monitored
        std::size_t size() const {
            return index_.size();
        }

        /// Data type used to store the crossings.
        /// Provided to make type-generic calling code.
        using crossing_list =  std::vector<threshold_crossing>;

    private:
        const_view values_;
        iarray index_;

        array thresholds_;
        value_type t_prev_;
        array v_prev_;
        crossing_list crossings_;
        iarray is_crossed_;
    };


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

