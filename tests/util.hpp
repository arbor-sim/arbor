#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static
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

