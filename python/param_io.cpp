#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <iomanip>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/primitives.hpp>
#include <sup/json_params.hpp>

#include "error.hpp"
#include "strprintf.hpp"

namespace pyarb {
    using sup::find_and_remove_json;

    arb::cable_cell_parameter_set load_cable_cell_parameter_set(nlohmann::json& params_json) {
        arb::cable_cell_parameter_set params;

        params.init_membrane_potential = find_and_remove_json<double>("Vm", params_json);
        params.membrane_capacitance    = find_and_remove_json<double>("cm", params_json);
        params.axial_resistivity       = find_and_remove_json<double>("Ra", params_json);
        if (auto temp_c = find_and_remove_json<double>("celsius", params_json)) {
            params.temperature_K = temp_c.value() + 273.15;
        }

        if (auto ions_json = find_and_remove_json<nlohmann::json>("ions", params_json)) {
            auto ions_map = ions_json.value().get<std::unordered_map<std::string, nlohmann::json>>();
            for (auto& i: ions_map) {
                auto ion_name = i.first;
                auto ion_json = i.second;

                arb::cable_cell_ion_data ion_data;
                if (auto iconc = find_and_remove_json<double>("internal-concentration", ion_json)) {
                    ion_data.init_int_concentration = iconc.value();
                }
                if (auto econc = find_and_remove_json<double>("external-concentration", ion_json)) {
                    ion_data.init_ext_concentration = econc.value();
                }
                if (auto rev_pot = find_and_remove_json<double>("reversal-potential", ion_json)) {
                    ion_data.init_reversal_potential = rev_pot.value();
                }
                params.ion_data.insert({ion_name, ion_data});

                if (auto method = find_and_remove_json<std::string>("method", ion_json)) {
                    if (method.value() == "nernst") {
                        params.reversal_potential_method.insert({ion_name, "nernst/" + ion_name});
                    } else if (method.value() != "constant") {
                        params.reversal_potential_method.insert({ion_name, method.value()});
                    }
                }
            }
        }
        return params;
    }

    arb::mechanism_desc load_mechanism_desc(nlohmann::json& mech_json) {
        auto name = find_and_remove_json<std::string>("mechanism", mech_json);
        if (name) {
            auto mech = arb::mechanism_desc(name.value());
            auto params = find_and_remove_json<std::unordered_map<std::string, double>>("parameters", mech_json);
            if (params) {
                for (auto p: params.value()) {
                    mech.set(p.first, p.second);
                }
            }
            return mech;
        }
        throw pyarb::pyarb_error("Mechanism not specified");
    }

    nlohmann::json make_cable_cell_parameter_set_json(const arb::cable_cell_parameter_set& params) {
        nlohmann::json record;
        if(auto tempK = params.temperature_K)        record["celsius"] = tempK.value() - 273.15;
        if(auto Vm = params.init_membrane_potential) record["Vm"] = Vm.value();
        if(auto Ra = params.axial_resistivity)       record["Ra"] = Ra.value();
        if(auto cm = params.membrane_capacitance)    record["cm"] = cm.value();
        for (auto ion: params.ion_data) {
            auto name = ion.first;
            auto data = ion.second;
            if(auto iconc = data.init_int_concentration) record["ions"][name]["internal-concentration"] = iconc.value();
            if(auto econc = data.init_ext_concentration) record["ions"][name]["external-concentration"] = econc.value();
            if(auto rvpot = data.init_reversal_potential) record["ions"][name]["reversal-potential"] = rvpot.value();
            if (params.reversal_potential_method.count(name)) {
                record["ions"][name]["method"] = params.reversal_potential_method.at(name).name();
            } else {
                record["ions"][name]["method"] = "constant";
            }
        }
        return record;
    }

    void store_cable_cell_parameter_set(const arb::cable_cell_parameter_set& set, std::string file_name) {
        std::ofstream file(file_name);
        file << std::setw(2) << make_cable_cell_parameter_set_json(set);
    };

    void store_decor(const arb::decor& decor, std::string file_name) {
        // Global
        nlohmann::json json_file;
        json_file["global"] = make_cable_cell_parameter_set_json(decor.defaults());

        // Local
        std::unordered_map<std::string, nlohmann::json> region_map;
        std::vector<nlohmann::json> mechs, regions;

        for (const auto& entry: decor.paintings()) {
            auto region_expr = to_string(entry.first);
            std::visit(
                    [&](auto&& p) {
                        using T = std::decay_t<decltype(p)>;
                        if constexpr (std::is_same_v<arb::init_membrane_potential, T>) {
                            region_map[region_expr]["Vm"] = p.value;
                        } else if constexpr (std::is_same_v<arb::axial_resistivity, T>) {
                            region_map[region_expr]["Ra"] = p.value;
                        } else if constexpr (std::is_same_v<arb::temperature_K, T>) {
                            region_map[region_expr]["celsius"] = p.value  - 273.15;
                        } else if constexpr (std::is_same_v<arb::membrane_capacitance, T>) {
                            region_map[region_expr]["cm"] = p.value;
                        } else if constexpr (std::is_same_v<arb::init_int_concentration, T>) {
                            region_map[region_expr][p.ion]["internal-concentration"] = p.value;
                        } else if constexpr (std::is_same_v<arb::init_ext_concentration, T>) {
                            region_map[region_expr][p.ion]["external-concentration"] = p.value;
                        } else if constexpr (std::is_same_v<arb::init_reversal_potential, T>) {
                            region_map[region_expr][p.ion]["reversal-potential"] = p.value;
                        } else if constexpr (std::is_same_v<arb::mechanism_desc, T>) {
                            nlohmann::json data;
                            data["region"] = region_expr;
                            data["mechanism"] = p.name();
                            data["parameters"] = p.values();
                            mechs.push_back(data);
                        }
                    },
                    entry.second);
        }
        for (auto reg: region_map) {
            reg.second["region"] = reg.first;
            regions.push_back(reg.second);
        }

        json_file["local"] = regions;
        json_file["mechanisms"] = mechs;

        std::ofstream file(file_name);
        file << std::setw(2) << json_file;
    }

    void register_param_loader(pybind11::module& m) {
        m.def("load_default_parameters",
              [](std::string fname) {
                  std::ifstream fid{fname};
                  if (!fid.good()) {
                      throw pyarb_error(util::pprintf("can't open file '{}'", fname));
                  }
                  nlohmann::json defaults_json;
                  fid >> defaults_json;
                  auto defaults = load_cable_cell_parameter_set(defaults_json);
                  try {
                      arb::cable_cell_global_properties G;
                      G.default_parameters = defaults;
                      arb::check_global_properties(G);
                  }
                  catch (std::exception& e) {
                      throw pyarb_error("error loading parameter from \"" + fname + "\": " + std::string(e.what()));
                  }
                  return defaults;
              },
              "Load default cell parameters from file.");

        m.def("load_decor",
              [](std::string fname) {
                  std::ifstream fid{fname};
                  if (!fid.good()) {
                      throw pyarb_error(util::pprintf("can't open file '{}'", fname));
                  }
                  nlohmann::json decor_json;
                  fid >> decor_json;
                  arb::decor decor;

                  // Global
                  auto globals_json = find_and_remove_json<nlohmann::json>("global", decor_json);
                  if (globals_json) {
                      auto defaults = load_cable_cell_parameter_set(globals_json.value());
                      decor.set_default(arb::membrane_capacitance{defaults.membrane_capacitance.value()});
                      decor.set_default(arb::axial_resistivity{defaults.axial_resistivity.value()});
                      decor.set_default(arb::temperature_K{defaults.temperature_K.value()});
                      decor.set_default(arb::init_membrane_potential{defaults.init_membrane_potential.value()});
                      for (auto ion: defaults.ion_data) {
                          decor.set_default(arb::initial_ion_data{ion.first, ion.second});
                      }
                      for (auto ion: defaults.reversal_potential_method) {
                          decor.set_default(arb::ion_reversal_potential_method{ion.first, ion.second});
                      }
                  }

                  // Local
                  auto locals_json = find_and_remove_json<std::vector<nlohmann::json>>("local", decor_json);
                  if (locals_json) {
                      for (auto l: locals_json.value()) {
                          auto region = find_and_remove_json<std::string>("region", l);
                          if (!region) {
                              throw pyarb_error("Local cell parameters do not include region label (in \"" + fname + "\")");
                          }
                          auto region_defaults = load_cable_cell_parameter_set(l);
                          if(!region_defaults.reversal_potential_method.empty()) {
                              throw pyarb_error("Cannot implement local reversal potential methods (in \"" + fname + "\")");
                          }

                          auto reg = region.value();
                          decor.paint(reg, arb::membrane_capacitance{region_defaults.membrane_capacitance.value()});
                          decor.paint(reg, arb::axial_resistivity{region_defaults.axial_resistivity.value()});
                          decor.paint(reg, arb::init_membrane_potential{region_defaults.init_membrane_potential.value()});
                          decor.paint(reg, arb::temperature_K{region_defaults.temperature_K.value()});
                          for (auto ion: region_defaults.ion_data) {
                              decor.paint(reg, arb::init_int_concentration{ion.first, ion.second.init_int_concentration.value()});
                              decor.paint(reg, arb::init_ext_concentration{ion.first, ion.second.init_ext_concentration.value()});
                              decor.paint(reg, arb::init_reversal_potential{ion.first, ion.second.init_reversal_potential.value()});
                          }
                      }
                  }

                  // Mechanisms
                  auto mech_json = find_and_remove_json<std::vector<nlohmann::json>>("mechanisms", decor_json);
                  if (mech_json) {
                      for (auto m: mech_json.value()) {
                          auto region = find_and_remove_json<std::string>("region", m);
                          if (!region) {
                              throw pyarb_error("Mechanisms do not include region label (in \"" + fname + "\")");
                          }
                          arb::mechanism_desc mech;
                          try {
                             mech = load_mechanism_desc(m);
                          }
                          catch (std::exception& e) {
                              throw pyarb_error("error loading mechanism for region " + region.value() + " in file \"" + fname + "\": " + std::string(e.what()));
                          }
                          decor.paint(region.value(), mech);
                      }
                  }
                  return decor;
              },
              "Load decor from file.");

        // arb::cable_cell_parameter_set
        pybind11::class_<arb::cable_cell_parameter_set> cable_cell_parameter_set(m, "cable_cell_parameter_set");
        cable_cell_parameter_set
                .def("__repr__", [](const arb::cable_cell_parameter_set& s) { return util::pprintf("<arbor.cable_cell_parameter_set>"); })
                .def("__str__", [](const arb::cable_cell_parameter_set& s) { return util::pprintf("(cell_parameter_set)"); });
    }

} //namespace pyarb