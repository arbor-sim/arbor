#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cmath>

#include <util.hpp>

[[gnu::unused]] static
void write_vis_file(const std::string& fname, std::vector<std::vector<double>> values)
{
    auto m = values.size();
    if(!m) return;

    std::ofstream fid(fname);
    if(!fid.is_open()) return;

    auto n = values[0].size();
    for(const auto& v : values) {
        if(n!=v.size()) {
            std::cerr << "all output arrays must have the same length\n";
            return;
        }
    }

    for(auto i=0u; i<n; ++i) {
        for(auto j=0u; j<m; ++j) {
            fid << " " << values[j][i];
        }
        fid << "\n";
    }
}

template <typename T>
std::vector<T> find_spikes(std::vector<T> const& v, T threshold, T dt)
{
    if(v.size()<2) {
        return {};
    }

    std::vector<T> times;
    for(auto i=1u; i<v.size(); ++i) {
        if(v[i]>=threshold && v[i-1]<threshold) {
            auto pos = (threshold-v[i-1]) / (v[i]-v[i-1]);
            times.push_back((i-1+pos)*dt);
        }
    }

    return times;
}

struct spike_comparison {
    double min = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
    double mean = std::numeric_limits<double>::quiet_NaN();
    double rms = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> diff;

    // check whether initialized (i.e. has valid results)
    bool is_valid() const {
        return min == min;
    }

    // return maximum relative error
    double max_relative_error() const {
        if(!is_valid()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return *std::max_element(diff.begin(), diff.end());
    }
};

std::ostream&
operator<< (std::ostream& o, spike_comparison const& spikes)
{
    return o << "[" << spikes.min << ", " << spikes.max << "], "
             << "mean " << spikes.mean << ", rms " << spikes.rms
             << ", diffs " << spikes.diff;
}

template <typename T>
spike_comparison compare_spikes(std::vector<T> const& spikes, std::vector<T> const& baseline)
{
    spike_comparison c;

    // return default initialized (all NaN) if number of spikes differs
    if(spikes.size() != baseline.size()) {
        return c;
    }

    c.min  = std::numeric_limits<double>::max();
    c.max  = 0.;
    c.mean = 0.;
    c.rms  = 0.;

    auto n = spikes.size();
    for(auto i=0u; i<n; ++i) {
        auto error = std::fabs(spikes[i] - baseline[i]);
        c.min = std::min(c.min, error);
        c.max = std::max(c.max, error);
        c.mean += error;
        c.rms += error*error;
        // relative difference
        c.diff.push_back(error/baseline[i]);
    }

    c.mean /= n;
    c.rms = std::sqrt(c.rms/n);

    return c;
}

