#include <fstream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <vector>

#include "arborio/jsonio.hpp"
#include "json_helpers.hpp"

namespace arborio {

jsonio_error::jsonio_error(const std::string& msg): arbor_exception(msg) {}

arb::cable_cell_parameter_set load_cable_cell_parameter_set(nlohmann::json& params_json) {
    arb::cable_cell_parameter_set params;

    params.init_membrane_potential = find_and_remove_json<double>("Vm", params_json);
    params.membrane_capacitance = find_and_remove_json<double>("cm", params_json);
    params.axial_resistivity = find_and_remove_json<double>("Ra", params_json);
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

arb::decor load_decor(nlohmann::json& decor_json) {
    arb::decor decor;

    // Global
    auto globals_json = find_and_remove_json<nlohmann::json>("global", decor_json);
    if (globals_json) {
        arb::cable_cell_parameter_set defaults;
        try {
            defaults = load_cable_cell_parameter_set(globals_json.value());
        } catch (std::exception& e) {
            throw jsonio_error("Error loading global parameters" + std::string(e.what()));
        }
        try {
            for (auto def: defaults.serialize()) {
                decor.set_default(def);
            }
        } catch (std::exception& e) {
            throw jsonio_error("Error setting global parameters" + std::string(e.what()));
        }
    }

    // Local
    auto locals_json = find_and_remove_json<std::vector<nlohmann::json>>("local", decor_json);
    if (locals_json) {
        for (auto l: locals_json.value()) {
            auto region = find_and_remove_json<std::string>("region", l);
            if (!region) {
                throw jsonio_error("Local cell parameters do not include region label");
            }
            std::string reg = region.value();

            // if the region expression does not start with an open parenthesis
            // it is not an s-expression, it is a label and we must add double quotes.
            if (reg.at(0) != '(') {
                reg = "\"" + region.value() + "\"";
            }

            arb::cable_cell_parameter_set region_defaults;
            try {
                region_defaults = load_cable_cell_parameter_set(l);
            } catch (std::exception& e) {
                throw jsonio_error("Error loading local parameters: " + std::string(e.what()));
            }

            if (!region_defaults.reversal_potential_method.empty()) {
                throw jsonio_error("Cannot implement regional reversal potential methods ");
            }
            try {
                if (auto v = region_defaults.membrane_capacitance) {
                    decor.paint(reg, arb::membrane_capacitance{v.value()});
                }
                if (auto v = region_defaults.axial_resistivity) {
                    decor.paint(reg, arb::axial_resistivity{v.value()});
                }
                if (auto v = region_defaults.init_membrane_potential) {
                    decor.paint(reg, arb::init_membrane_potential{v.value()});
                }
                if (auto v = region_defaults.temperature_K) {
                    decor.paint(reg, arb::temperature_K{v.value()});
                }
                for (auto ion: region_defaults.ion_data) {
                    if (auto v = ion.second.init_int_concentration) {
                        decor.paint(reg, arb::init_int_concentration{ion.first, v.value()});
                    }
                    if (auto v = ion.second.init_ext_concentration) {
                        decor.paint(reg, arb::init_ext_concentration{ion.first, v.value()});
                    }
                    if (auto v = ion.second.init_reversal_potential) {
                        decor.paint(reg, arb::init_reversal_potential{ion.first, v.value()});
                    }
                }
            } catch (std::exception& e) {
                throw jsonio_error("Error painting local parameters on region \"" + reg + "\": " + std::string(e.what()));
            }
        }
    }

    // Mechanisms
    auto mechs_json = find_and_remove_json<std::vector<nlohmann::json>>("mechanisms", decor_json);
    if (mechs_json) {
        for (auto mech_json: mechs_json.value()) {
            auto region = find_and_remove_json<std::string>("region", mech_json);
            if (!region) {
                throw jsonio_error("Mechanism description does not include region label");
            }
            auto name = find_and_remove_json<std::string>("mechanism", mech_json);
            if (!name) {
                throw jsonio_error("Mechanism description does not include mechanism name");
            }
            auto mech = arb::mechanism_desc(name.value());
            auto params = find_and_remove_json<std::unordered_map<std::string, double>>("parameters", mech_json);
            if (params) {
                for (auto p: params.value()) {
                    mech.set(p.first, p.second);
                }
            }
            auto reg = region.value();

            // if the region expression does not start with an open parenthesis
            // it is not an s-expression, it is a label and we must add double quotes.
            if (reg.at(0) != '(') {
                reg = "\"" + region.value() + "\"";
            }
            try {
                decor.paint(reg, mech);
            } catch (std::exception& e) {
                throw jsonio_error("Error painting mechanism \"" + name.value() + "\" on region \"" + reg + "\": " + std::string(e.what()));
            }
        }
    }
    return decor;
}

nlohmann::json make_cable_cell_parameter_set_json(const arb::cable_cell_parameter_set& params) {
    nlohmann::json record;
    if (auto tempK = params.temperature_K) record["celsius"] = tempK.value() - 273.15;
    if (auto Vm = params.init_membrane_potential) record["Vm"] = Vm.value();
    if (auto Ra = params.axial_resistivity) record["Ra"] = Ra.value();
    if (auto cm = params.membrane_capacitance) record["cm"] = cm.value();
    for (auto ion: params.ion_data) {
        auto name = ion.first;
        auto data = ion.second;
        if (auto iconc = data.init_int_concentration) record["ions"][name]["internal-concentration"] = iconc.value();
        if (auto econc = data.init_ext_concentration) record["ions"][name]["external-concentration"] = econc.value();
        if (auto rvpot = data.init_reversal_potential) record["ions"][name]["reversal-potential"] = rvpot.value();
        if (params.reversal_potential_method.count(name)) {
            record["ions"][name]["method"] = params.reversal_potential_method.at(name).name();
        } else {
            record["ions"][name]["method"] = "constant";
        }
    }
    return record;
}

nlohmann::json make_decor_json(const arb::decor& decor) {
    // Global
    nlohmann::json json_decor;
    json_decor["global"] = make_cable_cell_parameter_set_json(decor.defaults());

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
                        region_map[region_expr]["celsius"] = p.value - 273.15;
                    } else if constexpr (std::is_same_v<arb::membrane_capacitance, T>) {
                        region_map[region_expr]["cm"] = p.value;
                    } else if constexpr (std::is_same_v<arb::init_int_concentration, T>) {
                        region_map[region_expr]["ions"][p.ion]["internal-concentration"] = p.value;
                    } else if constexpr (std::is_same_v<arb::init_ext_concentration, T>) {
                        region_map[region_expr]["ions"][p.ion]["external-concentration"] = p.value;
                    } else if constexpr (std::is_same_v<arb::init_reversal_potential, T>) {
                        region_map[region_expr]["ions"][p.ion]["reversal-potential"] = p.value;
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

    json_decor["local"] = regions;
    json_decor["mechanisms"] = mechs;

    return  json_decor;
}

// Public functions - read and write directly from and to files

arb::cable_cell_parameter_set load_cable_cell_parameter_set(std::string fname) {
    std::ifstream fid{fname};
    if (!fid.good()) {
        throw jsonio_error("can't open file '{}'" + fname);
    }
    nlohmann::json defaults_json;
    fid >> defaults_json;
    try {
        return load_cable_cell_parameter_set(defaults_json);
    }
    catch (std::exception& e) {
        throw jsonio_error("Error loading cable_cell_parameter_set from \"" + fname + "\": " + std::string(e.what()));
    }
}

arb::decor load_decor(std::string fname) {
    std::ifstream fid{fname};
    if (!fid.good()) {
        throw jsonio_error("can't open file '{}'" + fname);
    }
    nlohmann::json decor_json;
    fid >> decor_json;
    try {
        return load_decor(decor_json);
    }
    catch (std::exception& e) {
        throw jsonio_error("Error loading decor from \"" + fname + "\": " + std::string(e.what()));
    }
    return load_decor(decor_json);
}

void store_cable_cell_parameter_set(const arb::cable_cell_parameter_set& set, std::string fname) {
    std::ofstream file(fname);
    file << std::setw(2) << make_cable_cell_parameter_set_json(set);
};

void store_decor(const arb::decor& decor, std::string fname) {
    std::ofstream file(fname);
    file << std::setw(2) << make_decor_json(decor);
}

} // namespace arborio