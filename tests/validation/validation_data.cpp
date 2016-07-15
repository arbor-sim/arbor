#include "validation_data.hpp"

#include <fstream>

namespace testing {

nlohmann::json data_loader::load(const std::string& name) const {
    std::string data_path=path_+'/'+name;
    std::ifstream fid(data_path);
    if (!fid) {
        throw std::runtime_error("unable to load validation data: "+data_path);
    }

    nlohmann::json data;
    fid >> data;
    return data;
}

data_loader g_validation_data;

} // namespace testing
