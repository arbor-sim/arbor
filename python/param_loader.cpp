#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/primitives.hpp>
#include <sup/json_params.hpp>

#include "error.hpp"

namespace pyarb {
using nlohmann::json;
using sup::param_from_json;

arb::cable_cell_parameter_set load_cell_defaults(std::istream& is) {
    double celsius, Vm, Ra, cm;
    arb::cable_cell_parameter_set defaults;

    json defaults_json, ions_json;
    defaults_json << is;

    try {
        param_from_json(ions_json, "ions", defaults_json);
        param_from_json(Vm, "V", defaults_json);
        param_from_json(cm, "cm", defaults_json);
        param_from_json(Ra, "Ra", defaults_json);
        param_from_json(celsius, "celsius", defaults_json);
    }
    catch (std::exception& e) {
        throw pyarb_error("necessary cell default value missing: " + std::string(e.what()));
    }

    auto ions_map = ions_json.get<std::unordered_map<std::string, nlohmann::json>>();
    for (auto& i: ions_map) {
        auto ion_name = i.first;
        auto ion_json = i.second;

        arb::cable_cell_ion_data ion_data;
        std::string method;

        try {
            param_from_json(ion_data.init_int_concentration, "internal-concentration", ion_json);
            param_from_json(ion_data.init_ext_concentration, "external-concentration", ion_json);
            param_from_json(ion_data.init_reversal_potential, "reversal-potential", ion_json);
            param_from_json(method, "method", ion_json);
        }
        catch (std::exception& e) {
            throw pyarb_error("necessary cell default for ion \"" + ion_name + "\" value missing: " + std::string(e.what()));
        }

        defaults.ion_data.insert({ion_name, ion_data});
        if(method == "nernst") {
            defaults.reversal_potential_method.insert({ion_name,"nernst/"+ion_name});
        } else if (method != "constant") {
            std::cout << "here " << method << std::endl;
            throw pyarb_error("method of ion \"" + ion_name + "\" can only be either constant or nernst");
        }
    }

    defaults.init_membrane_potential = Vm;
    defaults.membrane_capacitance = cm;
    defaults.axial_resistivity = Ra;
    defaults.temperature_K = celsius + 273.15;

    return defaults;
}

void register_param_loader(pybind11::module& m) {
    m.def("load_cell_defaults",
          [](std::string fname) {
              std::ifstream fid{fname};
              if (!fid.good()) {
                  throw pyarb_error(util::pprintf("can't open file '{}'", fname));
              }
              try {
                  return load_cell_defaults(fid);
              }
              catch (std::exception& e) {
                  throw pyarb_error("error loading parameter from \"" + fname + "\": " + std::string(e.what()));
              }
          },
          "Load default cel parameters.");

    // arb::cable_cell_parameter_set
    pybind11::class_<arb::cable_cell_parameter_set> cable_cell_parameter_set(m, "cable_cell_parameter_set");
}

} //namespace pyarb