#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
    bool is_spiking = v[0] >= threshold;
    auto it = v.begin() + 1;

    std::vector<T> times;
    for(auto i=1u; i<v.size(); ++i) {
        if(is_spiking && v[i]<threshold) {
            is_spiking = false;
        }
        else {
            if(*it>=threshold) {
                is_spiking = true;
                auto pos = (threshold-v[i-1]) / (v[i]-v[i-1]);
                times.push_back((i-1+pos)*dt);
            }
        }
    }


    return times;
}

