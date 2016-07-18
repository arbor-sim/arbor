#pragma once

#include <json/src/json.hpp>

#ifndef DATADIR
#define DATADIR "../data"
#endif

namespace testing {

class data_loader {
public:
    // set where to find the validation JSON files
    void set_path(const std::string& path) { path_=path; }

    // load JSON file from path plus name, throw exception
    // if cannot be found or loaded.
    nlohmann::json load(const std::string& name) const;

private:
    std::string path_=DATADIR "/validation";
};

extern data_loader g_validation_data;

} // namespace testing
