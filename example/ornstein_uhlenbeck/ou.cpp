#include <fmt/format.h>

#include "ornstein_uhlenbeck_catalogue.hpp"

#include <arborio/label_parse.hpp>

#include <arbor/assert.hpp>
#include <arbor/recipe.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/simulation.hpp>
#include <arbor/sampling.hpp>
#include <arbor/common_types.hpp>

// a single-cell recipe with probes
struct recipe: public arb::recipe {
    recipe(unsigned ncvs) {
        using namespace arb;
        using namespace arborio::literals;

        // build catalogue with stochastic mechanism
        cell_gprop_.catalogue = global_default_catalogue();
        cell_gprop_.catalogue.extend(arb::global_ornstein_uhlenbeck_catalogue());
        cell_gprop_.default_parameters = neuron_parameter_defaults;
        
        // paint the process on the whole cell
        decor dec;
        double const cv_size = 1.0;
        dec.paint("(all)"_reg , density("hh"));
        dec.paint("(all)"_reg , density("ornstein_uhlenbeck"));

        // single-cell tree with ncvs control volumes
        segment_tree tree;
        tree.append(mnpos, {0, 0, 0.0, 4.0}, {0, 0, ncvs*cv_size, 4.0}, 1);
        cell_ = cable_cell(morphology(tree), dec, {}, cv_policy_max_extent(cv_size));
    }

    arb::cell_size_type num_cells() const override { return 1; }

    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override { return cell_; }

    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::cable; }

    std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override { return probes_; }

    std::any get_global_properties(arb::cell_kind) const override { return cell_gprop_; }

    void add_probe(const arb::cell_tag_type& tag, std::any address) { probes_.emplace_back(std::move(address), tag); }

protected:
    std::vector<arb::probe_info> probes_;
    arb::cable_cell_global_properties cell_gprop_;
    arb::cable_cell cell_;
};

// sampler for vector probes
struct sampler {
    sampler(std::vector<arb_value_type>& data, std::size_t n_cvs, std::size_t n_steps):
     n_cvs_{n_cvs} , n_steps_{n_steps}, data_{data} {
        data_.resize(n_cvs*n_steps);
    }

    void operator()(arb::probe_metadata pm, const arb::sample_records& samples) {
        auto reader = arb::sample_reader<arb::cable_state_cell_meta_type>(pm.meta, samples);
        auto width = reader.n_column();
        for (std::size_t ix = 0; ix < reader.n_row(); ++ix) {
            for (std::size_t iy = 0; iy < reader.n_column(); ++iy) {
                data_[ix*width + iy] = reader.value(ix, iy);
            }
        }
    }

    std::size_t n_cvs_;
    std::size_t n_steps_;
    std::vector<arb_value_type>& data_;
};

// compute mean and variance online
// uses Welford-Knuth algorithm for the variance
struct accumulator {
    std::size_t n_ = 0;
    double mean_ = 0;
    double var_ = 0;

    accumulator& operator()(double sample) {
        double const delta = sample - mean_;
        mean_ += delta / (++n_);
        var_ += delta * (sample - mean_);
        return *this;
    }

    std::size_t n() const noexcept { return n_; }
    double mean() const noexcept { return mean_; }
    double variance() const noexcept { return n_ > 1 ? var_/(n_-1) : 0; }
};

int main(int argc, char** argv) {

    unsigned ncvs = 5000;        // number of control volumes
    double const dt = 1.0/1024;   // time step
    unsigned nsteps = 500;       // number of time steps

    // create recipe and add probes
    recipe rec(ncvs);
    rec.add_probe("ou-S", arb::cable_probe_density_state_cell{"ornstein_uhlenbeck", "S"});

    // make context and simulation objects
    auto context = arb::make_context({1, -1});
    arb::simulation sim = arb::simulation::create(rec)
        .set_context(context)
        .set_seed(137);

    // setup sampler and add it to the simulation with regular schedule
    std::vector<arb_value_type> data;
    sampler s{data, ncvs, nsteps};
    sim.add_sampler(arb::all_probes, arb::regular_schedule(dt*arb::units::ms), s);

    // run the simulation
    sim.run(nsteps*dt*arb::units::ms, dt*arb::units::ms);

    // evaluate the mean for each time step across the ensembe of realizations
    // (each control volume is a independent realization of the Ornstein-Uhlenbeck process)
    std::vector<accumulator> acc(nsteps);
    for (std::size_t t=0; t<nsteps; ++t) {
        for (std::size_t i=0; i<ncvs; ++i) {
            acc[t](s.data_[t*ncvs+ i]);
        }
    }

    // analytical solutions
    auto expected = [](double t) -> std::pair<double,double> {
        const double mu = 1.0;
        const double S_0 = 2.0;
        const double kappa = 0.1;
        const double sigma = 0.1;
        return {
            mu - (mu-S_0)*std::exp(-kappa*t),
            (sigma*sigma/(2*kappa))*(1.0 - std::exp(-2*kappa*t))
        };
    };

    // print mean and expectation
    for (std::size_t t=0; t<nsteps; t+=10) {
        fmt::print("time = {:.5f}: mean = {:.5f} expected = {:.5f}\n",
            dt*t, acc[t].mean(), expected(dt*t).first);
    }
    return 0;
}
