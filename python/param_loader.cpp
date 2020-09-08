#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/primitives.hpp>

#include "json_params.hpp"
#include "error.hpp"

namespace pyarb {
using nlohmann::json;
using sup::param_from_json;

arb::cable_cell_parameter_set load_cell_defaults(std::istream& is) {
    double celsius, Vm, Ra, cm;
    arb::cable_cell_parameter_set defaults;

    json defaults_json, ions_json;
    defaults_json << is;

    param_from_json(ions_json, "ions", defaults_json);
    auto ions_map = ions_json.get<std::unordered_map<std::string, nlohmann::json>>();
    for (auto& i: ions_map) {
        auto ion_name = i.first;
        auto ion_json = i.second;

        arb::cable_cell_ion_data ion_data;
        param_from_json(ion_data.init_int_concentration,  "internal-concentration", ion_json);
        param_from_json(ion_data.init_ext_concentration,  "external-concentration", ion_json);
        param_from_json(ion_data.init_reversal_potential, "reversal-potential", ion_json);
        defaults.ion_data.insert({ion_name, ion_data});

        std::string method;
        param_from_json(method, "method", ion_json);
        if(method == "nernst") {
            defaults.reversal_potential_method.insert({ion_name,"nernst/"+ion_name});
        } else if (method != "constant") {
            std::cout << "here " << method << std::endl;
            throw pyarb_error("method of ion \"" + ion_name + "\" can only be either constant or nernst");
        }
    }

    param_from_json(Vm, "Vm", defaults_json);
    param_from_json(cm, "cm", defaults_json);
    param_from_json(Ra, "Ra", defaults_json);
    param_from_json(celsius, "celsius", defaults_json);

    defaults.init_membrane_potential = Vm;
    defaults.membrane_capacitance = cm;
    defaults.axial_resistivity = Ra;
    defaults.temperature_K = celsius + 273.15;

    std::cout << "Vm " << defaults.init_membrane_potential.value() << std::endl;
    std::cout << "cm " << defaults.membrane_capacitance.value() << std::endl;
    std::cout << "Ra " << defaults.axial_resistivity.value() << std::endl;
    std::cout << "temp " << defaults.temperature_K.value() << std::endl;

    std::cout << "ca_rev_pot "  << defaults.ion_data["ca"].init_reversal_potential<< std::endl;
    std::cout << "ca_int_conc " << defaults.ion_data["ca"].init_int_concentration<< std::endl;
    std::cout << "ca_ext_conc " << defaults.ion_data["ca"].init_ext_concentration<< std::endl;
    if(defaults.reversal_potential_method.count("ca")) std::cout << "ca_method "   << defaults.reversal_potential_method["ca"].name() << std::endl;

    std::cout << "na_rev_pot "  << defaults.ion_data["na"].init_reversal_potential<< std::endl;
    std::cout << "na_int_conc " << defaults.ion_data["na"].init_int_concentration<< std::endl;
    std::cout << "na_ext_conc " << defaults.ion_data["na"].init_ext_concentration<< std::endl;
    if(defaults.reversal_potential_method.count("na")) std::cout << "na_method "   << defaults.reversal_potential_method["na"].name() << std::endl;

    std::cout << "k_rev_pot "  << defaults.ion_data["k"].init_reversal_potential<< std::endl;
    std::cout << "k_int_conc " << defaults.ion_data["k"].init_int_concentration<< std::endl;
    std::cout << "k_ext_conc " << defaults.ion_data["k"].init_ext_concentration<< std::endl;
    if(defaults.reversal_potential_method.count("k")) std::cout << "k_method "   << defaults.reversal_potential_method["k   "].name() << std::endl;

    return defaults;
}

void register_param_loader(pybind11::module& m) {
    m.def("load_cell_defaults",
          [](std::string fname) {
              std::ifstream fid{fname};
              if (!fid.good()) {
                  throw pyarb_error(util::pprintf("can't open file '{}'", fname));
              }

              return load_cell_defaults(fid);
          },
          "Load default cel parameters.");
}

} //namespace pyarb