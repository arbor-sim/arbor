#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "arbor/util/any_visitor.hpp"
#include "arborio/jsonio.hpp"
#include "json_helpers.hpp"

namespace arborio {

jsonio_error::jsonio_error(const std::string& msg):
    arbor_exception(msg) {}

jsonio_unused_input::jsonio_unused_input(const std::string& key):
    jsonio_error("Unused input parameter: \"" + key + "\"")
{}

jsonio_decor_global_load_error::jsonio_decor_global_load_error(const std::string err):
    jsonio_error("Decor: error loading global parameters: " + err)
{}

jsonio_decor_global_set_error::jsonio_decor_global_set_error(const std::string err):
    jsonio_error("Decor: error setting decor global parameters: " + err)
{}

jsonio_decor_local_missing_region::jsonio_decor_local_missing_region():
    jsonio_error("Decor: regional parameters must include region label")
{}

jsonio_decor_local_revpot_mech::jsonio_decor_local_revpot_mech(const std::string& reg, const std::string& ion, const std::string& mech):
    jsonio_error("Decor: cannot implement regional reversal potential methods: \"" + reg + "\", (\"" + ion + "\", \"" + mech + "\")")
{}

jsonio_decor_local_load_error::jsonio_decor_local_load_error(const std::string err):
    jsonio_error("Decor: error loading decor regional parameters: " + err)
{}

jsonio_decor_local_set_error::jsonio_decor_local_set_error(const std::string err):
    jsonio_error("Decor: error painting regional parameters: " + err)
{}

jsonio_decor_mech_missing_region::jsonio_decor_mech_missing_region():
    jsonio_error("Decor: mechanism description does not include region label")
{}

jsonio_decor_mech_missing_name::jsonio_decor_mech_missing_name():
    jsonio_error("Decor: mechanism description does not include mechanism name")
{}

jsonio_decor_mech_set_error::jsonio_decor_mech_set_error(const std::string& reg, const std::string& mech, const std::string& err):
    jsonio_error("Decor: Error painting mechanism \"" + mech + "\" on region \"" + reg + "\": " + err)
{}

jsonio_json_parse_error::jsonio_json_parse_error(const std::string err):
    jsonio_error("Error parsing json : " + err)
{}

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
                }
                else if (method.value() != "constant") {
                    params.reversal_potential_method.insert({ion_name, method.value()});
                }
            }
        }
    }
    if (!params_json.empty()) {
        throw jsonio_unused_input(params_json.begin().key());
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
        }
        catch (std::exception& e) {
            throw jsonio_decor_global_load_error(e.what());
        }
        try {
            for (auto def: defaults.serialize()) {
                decor.set_default(def);
            }
        }
        catch (std::exception& e) {
            throw jsonio_decor_global_set_error(e.what());
        }
    }

    // Local
    auto locals_json = find_and_remove_json<std::vector<nlohmann::json>>("local", decor_json);
    if (locals_json) {
        for (auto l: locals_json.value()) {
            auto region = find_and_remove_json<std::string>("region", l);
            if (!region) {
                throw jsonio_decor_local_missing_region();
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
            }
            catch (std::exception& e) {
                throw jsonio_decor_local_load_error(e.what());
            }

            if (!region_defaults.reversal_potential_method.empty()) {
                auto rvpot_method = region_defaults.reversal_potential_method.begin();
                throw jsonio_decor_local_revpot_mech(reg, rvpot_method->first, rvpot_method->second.name());
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
            }
            catch (std::exception& e) {
                throw jsonio_decor_local_set_error(e.what());
            }
        }
    }

    // Mechanisms
    auto mechs_json = find_and_remove_json<std::vector<nlohmann::json>>("mechanisms", decor_json);
    if (mechs_json) {
        for (auto mech_json: mechs_json.value()) {
            auto region = find_and_remove_json<std::string>("region", mech_json);
            if (!region) {
                throw jsonio_decor_mech_missing_region();
            }
            auto name = find_and_remove_json<std::string>("mechanism", mech_json);
            if (!name) {
                throw jsonio_decor_mech_missing_name();
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
            }
            catch (std::exception& e) {
                throw jsonio_decor_mech_set_error(reg, name.value(), e.what());
            }
        }
    }
    if (!decor_json.empty()) {
        throw jsonio_unused_input(decor_json.begin().key());
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
        }
        else {
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
    std::map<std::string, nlohmann::json> region_map;
    std::vector<nlohmann::json> mechs, regions;

    for (const auto& entry: decor.paintings()) {
        auto region_expr = to_string(entry.first);

        auto paintable_visitor = arb::util::overload(
            [&](const arb::init_membrane_potential& p) { region_map[region_expr]["Vm"] = p.value; },
            [&](const arb::axial_resistivity& p)       { region_map[region_expr]["Ra"] = p.value; },
            [&](const arb::temperature_K& p)           { region_map[region_expr]["celsius"] = p.value - 273.15; },
            [&](const arb::membrane_capacitance& p)    { region_map[region_expr]["cm"] = p.value; },
            [&](const arb::init_int_concentration& p)  { region_map[region_expr]["ions"][p.ion]["internal-concentration"] = p.value; },
            [&](const arb::init_ext_concentration& p)  { region_map[region_expr]["ions"][p.ion]["external-concentration"] = p.value; },
            [&](const arb::init_reversal_potential& p) { region_map[region_expr]["ions"][p.ion]["reversal-potential"] = p.value; },
            [&](const arb::mechanism_desc& p) {
                nlohmann::json data;
                data["region"] = region_expr;
                data["mechanism"] = p.name();
                if (!p.values().empty()) {
                    data["parameters"] = p.values();
                }
                mechs.push_back(data);
            });

        std::visit(paintable_visitor, entry.second);
    }
    for (auto reg: region_map) {
        reg.second["region"] = reg.first;
        regions.push_back(reg.second);
    }

    json_decor["local"] = regions;
    json_decor["mechanisms"] = mechs;

    return json_decor;
}

// Public functions - read and write directly from and to files

arb::cable_cell_parameter_set load_cable_cell_parameter_set(std::istream& s) {
    nlohmann::json defaults_json;
    try {
        s >> defaults_json;
    }
    catch (std::exception& e) {
        throw jsonio_json_parse_error(e.what());
    }
    return load_cable_cell_parameter_set(defaults_json);
}

arb::decor load_decor(std::istream& s) {
    nlohmann::json decor_json;
    try {
        s >> decor_json;
    }
    catch (std::exception& e) {
        throw jsonio_json_parse_error(e.what());
    }
    return load_decor(decor_json);
}

void store_cable_cell_parameter_set(const arb::cable_cell_parameter_set& set, std::ostream& s) {
    nlohmann::json json_set;
    json_set = make_cable_cell_parameter_set_json(set);
    s << std::setw(2) << json_set;
};

void store_decor(const arb::decor& decor, std::ostream& s) {
    nlohmann::json json_decor;
    json_decor = make_decor_json(decor);
    s << std::setw(2) << json_decor;
}

} // namespace arborio