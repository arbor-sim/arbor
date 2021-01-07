#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/primitives.hpp>
#include <sup/json_params.hpp>

#include "error.hpp"

namespace pyarb {
    using sup::find_and_remove_json;

    arb::cable_cell_parameter_set load_cell_parameters(nlohmann::json& defaults_json) {
        arb::cable_cell_parameter_set defaults;

        defaults.init_membrane_potential = find_and_remove_json<double>("Vm", defaults_json);
        defaults.membrane_capacitance    = find_and_remove_json<double>("cm", defaults_json);
        defaults.axial_resistivity       = find_and_remove_json<double>("Ra", defaults_json);
        if (auto temp_c = find_and_remove_json<double>("celsius", defaults_json)) {
            defaults.temperature_K = temp_c.value() + 273.15;
        }

        if (auto ions_json = find_and_remove_json<nlohmann::json>("ions", defaults_json)) {
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
                defaults.ion_data.insert({ion_name, ion_data});

                if (auto method = find_and_remove_json<std::string>("method", ion_json)) {
                    if (method.value() == "nernst") {
                        defaults.reversal_potential_method.insert({ion_name, "nernst/" + ion_name});
                    } else if (method.value() != "constant") {
                        defaults.reversal_potential_method.insert({ion_name, method.value()});
                    }
                }
            }
        }
        return defaults;
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
    nlohmann::json create_param_set_json(const arb::cable_cell_parameter_set& mod) {
        nlohmann::json record;
        if(auto tempK = mod.temperature_K)        record["celsius"] = tempK.value() - 273.15;
        if(auto Vm = mod.init_membrane_potential) record["Vm"] = Vm.value();
        if(auto Ra = mod.axial_resistivity)       record["Ra"] = Ra.value();
        if(auto cm = mod.membrane_capacitance)    record["cm"] = cm.value();
        for (auto ion: mod.ion_data) {
            auto name = ion.first;
            auto data = ion.second;
            if(auto iconc = data.init_int_concentration) record["ions"][name]["internal-concentration"] = iconc.value();
            if(auto econc = data.init_ext_concentration) record["ions"][name]["external-concentration"] = econc.value();
            if(auto rvpot = data.init_reversal_potential) record["ions"][name]["reversal-potential"] = rvpot.value();
            if (mod.reversal_potential_method.count(name)) {
                record["ions"][name]["method"] = mod.reversal_potential_method.at(name).name();
            } else {
                record["ions"][name]["method"] = "constant";
            }
        }
        return record;
    }

    void output_parameter_set(const arb::cable_cell_parameter_set& set, std::string file_name) {
        std::ofstream file(file_name);
        file << std::setw(2) << create_param_set_json(set);
    };

    void output_cell_description(const arb::cable_cell& cell, std::string file_name) {
        // Global
        nlohmann::json json_file;
        json_file["global"] = create_param_set_json(cell.default_parameters);

        // Local
        std::unordered_map<std::string, nlohmann::json> regions;

        for (const auto& entry: cell.get_region_temperatures()) {
            regions[entry.first]["celsius"] = entry.second.value;
        }
        for (const auto& entry: cell.get_region_init_membrabe_potentials()) {
            regions[entry.first]["Vm"] = entry.second.value;
        }
        for (const auto& entry: cell.get_region_axial_resistivity()) {
            regions[entry.first]["Ra"] = entry.second.value;
        }
        for (const auto& entry: cell.get_region_membrane_capacitance()) {
            regions[entry.first]["cm"] = entry.second.value;
        }
        for (const auto& entry: cell.get_region_init_int_concentration()) {
            for (auto iconc: entry.second) {
                regions[entry.first]["ions"][iconc.ion]["internal-concentration"] = iconc.value;
            }
        }
        for (const auto& entry: cell.get_region_init_ext_concentration()) {
            for (auto econc: entry.second) {
                regions[entry.first]["ions"][econc.ion]["external-concentration"] = econc.value;
            }
        }
        for (const auto& entry: cell.get_region_init_reversal_potential()) {
            for (auto rvpot: entry.second) {
                regions[entry.first]["ions"][rvpot.ion]["reversal-potential"] = rvpot.value;
            }
        }

        std::vector<nlohmann::json> reg_vec;
        for (auto reg: regions) {
            reg.second["region"] = reg.first;
            reg_vec.push_back(reg.second);
        }

        json_file["local"] = reg_vec;

        // Mechs
        std::vector<nlohmann::json> mechs;
        for (const auto& entry: cell.get_region_mechanism_desc()) {
            auto reg = entry.first;
            for (auto& mech_desc: entry.second) {
                nlohmann::json data;
                data["region"] = reg;
                data["mechanism"] = mech_desc.name();
                data["parameters"] = mech_desc.values();
                mechs.push_back(data);
            }
        }

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
                  defaults_json << fid;
                  auto defaults = load_cell_parameters(defaults_json);
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
              "Load default cell parameters.");

        m.def("load_cell_parameters",
              [](std::string fname) {
                  std::ifstream fid{fname};
                  if (!fid.good()) {
                      throw pyarb_error(util::pprintf("can't open file '{}'", fname));
                  }
                  nlohmann::json cells_json;
                  cells_json << fid;
                  auto globals_json = find_and_remove_json<nlohmann::json>("global", cells_json);
                  if (globals_json) {
                      return load_cell_parameters(globals_json.value());
                  }
                  return arb::cable_cell_parameter_set();
              },
              "Load global cell parameters.");

        m.def("load_region_parameters",
              [](std::string fname) {
                  std::unordered_map<std::string, arb::cable_cell_parameter_set> local_map;

                  std::ifstream fid{fname};
                  if (!fid.good()) {
                      throw pyarb_error(util::pprintf("can't open file '{}'", fname));
                  }
                  nlohmann::json cells_json;
                  cells_json << fid;

                  auto locals_json = find_and_remove_json<std::vector<nlohmann::json>>("local", cells_json);
                  if (locals_json) {
                      for (auto l: locals_json.value()) {
                          auto region = find_and_remove_json<std::string>("region", l);
                          if (!region) {
                              throw pyarb_error("Local cell parameters do not include region label (in \"" + fname + "\")");
                          }
                          auto region_params = load_cell_parameters(l);

                          if(!region_params.reversal_potential_method.empty()) {
                              throw pyarb_error("Cannot implement local reversal potential methods (in \"" + fname + "\")");
                          }

                          local_map[region.value()] = region_params;
                      }
                  }
                  return local_map;
              },
              "Load local cell parameters.");

        m.def("load_region_mechanisms",
              [](std::string fname) {
                  std::unordered_map<std::string, std::vector<arb::mechanism_desc>> mech_map;

                  std::ifstream fid{fname};
                  if (!fid.good()) {
                      throw pyarb_error(util::pprintf("can't open file '{}'", fname));
                  }
                  nlohmann::json cells_json;
                  cells_json << fid;

                  auto mech_json = find_and_remove_json<std::vector<nlohmann::json>>("mechanisms", cells_json);
                  if (mech_json) {
                      for (auto m: mech_json.value()) {
                          auto region = find_and_remove_json<std::string>("region", m);
                          if (!region) {
                              throw pyarb_error("Mechanisms do not include region label (in \"" + fname + "\")");
                          }
                          try {
                              mech_map[region.value()].push_back(load_mechanism_desc(m));
                          }
                          catch (std::exception& e) {
                              throw pyarb_error("error loading mechanism for region " + region.value() + " in file \"" + fname + "\": " + std::string(e.what()));
                          }
                      }
                  }
                  return mech_map;
              },
              "Load region mechanism descriptions.");

        // arb::cable_cell_parameter_set
        pybind11::class_<arb::cable_cell_parameter_set> cable_cell_parameter_set(m, "cable_cell_parameter_set");
        cable_cell_parameter_set
                .def("__repr__", [](const arb::cable_cell_parameter_set& s) { return util::pprintf("<arbor.cable_cell_parameter_set>"); })
                .def("__str__", [](const arb::cable_cell_parameter_set& s) { return util::pprintf("(cell_parameter_set)"); });


        // map of arb::cable_cell_parameter_set
        pybind11::class_<std::unordered_map<std::string, arb::cable_cell_parameter_set>> region_parameter_map(m, "region_parameter_map");

        // map of arb::cable_cell_parameter_set
        pybind11::class_<std::unordered_map<std::string, std::vector<arb::mechanism_desc>>> region_mechanism_map(m, "region_mechanism_map");
    }

} //namespace pyarb