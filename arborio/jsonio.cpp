#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "arbor/util/any_visitor.hpp"
#include "arborio/jsonio.hpp"
#include "json_helpers.hpp"

namespace arb {
void to_json(nlohmann::json& j, const cable_cell_parameter_set& params) {
    if (auto tempK = params.temperature_K) j["temperature-K"] = tempK.value();
    if (auto Vm = params.init_membrane_potential) j["init-membrane-potential"] = Vm.value();
    if (auto Ra = params.axial_resistivity) j["axial-resistivity"] = Ra.value();
    if (auto cm = params.membrane_capacitance) j["membrane-capacitance"] = cm.value();
    for (auto ion: params.ion_data) {
        auto name = ion.first;
        auto data = ion.second;
        if (auto iconc = data.init_int_concentration) j["ions"][name]["internal-concentration"] = iconc.value();
        if (auto econc = data.init_ext_concentration) j["ions"][name]["external-concentration"] = econc.value();
        if (auto rvpot = data.init_reversal_potential) j["ions"][name]["reversal-potential"] = rvpot.value();
        if (params.reversal_potential_method.count(name)) {
            j["ions"][name]["method"] = params.reversal_potential_method.at(name).name();
        }
        else {
            j["ions"][name]["method"] = "constant";
        }
    }
}

void from_json(const nlohmann::json& j, cable_cell_parameter_set& params) {
    auto j_copy = j;
    params.init_membrane_potential = find_and_remove_json<double>("init-membrane-potential", j_copy);
    params.membrane_capacitance = find_and_remove_json<double>("membrane-capacitance", j_copy);
    params.axial_resistivity = find_and_remove_json<double>("axial-resistivity", j_copy);
    params.temperature_K = find_and_remove_json<double>("temperature-K", j_copy);

    if (auto ions_json = find_and_remove_json<nlohmann::json>("ions", j_copy)) {
        auto ions_map = ions_json.value().get<std::unordered_map<std::string, nlohmann::json>>();
        for (auto& [ion_name, ion_json]: ions_map) {
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
    if (!j_copy.empty()) {
        throw arborio::jsonio_unused_input(j_copy.begin().key());
    }
}

void to_json(nlohmann::json& j, const decor& decor) {
    // Global
    j["global"] = decor.defaults();

    // Local - order of insertion is important, concatenate regional (non-mechanism) paintings into 1 JSON instance when consecutive.
    std::vector<nlohmann::json> region_vec;

    nlohmann::json region;
    for (auto it = decor.paintings().begin(); it != decor.paintings().end(); ++it) {
        auto region_expr = to_string(it->first);

        auto paintable_visitor = arb::util::overload(
            [&](const arb::init_membrane_potential& p) { region["init-membrane-potential"] = p.value; },
            [&](const arb::axial_resistivity& p)       { region["axial-resistivity"] = p.value; },
            [&](const arb::temperature_K& p)           { region["temperature-K"] = p.value; },
            [&](const arb::membrane_capacitance& p)    { region["membrane-capacitance"] = p.value; },
            [&](const arb::init_int_concentration& p)  { region["ions"][p.ion]["internal-concentration"] = p.value; },
            [&](const arb::init_ext_concentration& p)  { region["ions"][p.ion]["external-concentration"] = p.value; },
            [&](const arb::init_reversal_potential& p) { region["ions"][p.ion]["reversal-potential"] = p.value; },
            [&](const arb::mechanism_desc& p) {
              nlohmann::json mech;
              mech["region"] = region_expr;
              mech["mechanism"] = p.name();
              if (!p.values().empty()) {
                  mech["parameters"] = p.values();
              }
              j["mechanisms"].push_back(mech);
            });

        std::visit(paintable_visitor, it->second);

        if (region.empty()) continue;
        auto it_next = it+1;
        if (it == decor.paintings().end()-1 || region_expr != to_string(it_next->first)) {
            region["region"] = region_expr;
            region_vec.push_back(region);
            region.clear();
        }
    }
    j["local"] = region_vec;
}

void from_json(const nlohmann::json& j, decor& decor) {
    auto j_copy = j;
    // Global
    if (auto globals_json = find_and_remove_json<nlohmann::json>("global", j_copy)) {
        arb::cable_cell_parameter_set defaults;
        try {
            defaults = globals_json.value().get<arb::cable_cell_parameter_set>();
        }
        catch (std::exception& e) {
            throw arborio::jsonio_decor_global_load_error(e.what());
        }
        try {
            for (auto def: defaults.serialize()) {
                decor.set_default(def);
            }
        }
        catch (std::exception& e) {
            throw arborio::jsonio_decor_global_set_error(e.what());
        }
    }

    // Local
    if (auto locals_json = find_and_remove_json<std::vector<nlohmann::json>>("local", j_copy)) {
        for (auto l: locals_json.value()) {
            auto region = find_and_remove_json<std::string>("region", l);
            if (!region) {
                throw arborio::jsonio_decor_local_missing_region();
            }
            std::string reg = region.value();
            arb::cable_cell_parameter_set region_defaults;
            try {
                region_defaults = l.get<arb::cable_cell_parameter_set>();
            }
            catch (std::exception& e) {
                throw arborio::jsonio_decor_local_load_error(e.what());
            }

            if (!region_defaults.reversal_potential_method.empty()) {
                throw arborio::jsonio_decor_local_revpot_mech();
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
                throw arborio::jsonio_decor_local_set_error(e.what());
            }
        }
    }

    // Mechanisms
    if (auto mechs_json = find_and_remove_json<std::vector<nlohmann::json>>("mechanisms", j_copy)) {
        for (auto mech_json: mechs_json.value()) {
            auto region = find_and_remove_json<std::string>("region", mech_json);
            if (!region) {
                throw arborio::jsonio_decor_mech_missing_region();
            }
            auto name = find_and_remove_json<std::string>("mechanism", mech_json);
            if (!name) {
                throw arborio::jsonio_decor_mech_missing_name();
            }
            auto mech = arb::mechanism_desc(name.value());
            auto params = find_and_remove_json<std::unordered_map<std::string, double>>("parameters", mech_json);
            if (params) {
                for (auto p: params.value()) {
                    mech.set(p.first, p.second);
                }
            }
            auto reg = region.value();
            try {
                decor.paint(reg, mech);
            }
            catch (std::exception& e) {
                throw arborio::jsonio_decor_mech_set_error(reg, name.value(), e.what());
            }
        }
    }
    if (!j_copy.empty()) {
        throw arborio::jsonio_unused_input(j_copy.begin().key());
    }
}
};
// namespace arb
namespace arborio {

jsonio_error::jsonio_error(const std::string& msg):
    arbor_exception(msg) {}

jsonio_json_parse_error::jsonio_json_parse_error(const std::string& err):
    jsonio_error("Error parsing JSON : " + err)
{}

jsonio_unused_input::jsonio_unused_input(const std::string& key):
    jsonio_error("Unused input parameter: \"" + key + "\"")
{}

jsonio_decor_global_load_error::jsonio_decor_global_load_error(const std::string& err):
    jsonio_error("Decor: error loading global parameters: " + err)
{}

jsonio_decor_global_set_error::jsonio_decor_global_set_error(const std::string& err):
    jsonio_error("Decor: error setting global parameters: " + err)
{}

jsonio_decor_local_missing_region::jsonio_decor_local_missing_region():
    jsonio_error("Decor: local parameters must include region label")
{}

jsonio_decor_local_revpot_mech::jsonio_decor_local_revpot_mech():
    jsonio_error("Decor: cannot implement local reversal potential methods")
{}

jsonio_decor_local_load_error::jsonio_decor_local_load_error(const std::string& err):
    jsonio_error("Decor: error loading local parameters: " + err)
{}

jsonio_decor_local_set_error::jsonio_decor_local_set_error(const std::string& err):
    jsonio_error("Decor: error painting local parameters: " + err)
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

jsonio_missing_field::jsonio_missing_field(const std::string& field):
    jsonio_error("Missing \"" + field + "\" field.")
{}

jsonio_version_error::jsonio_version_error(const std::string& ver):
    jsonio_error("Unsupported version: \"" + ver + "\".")
{}

jsonio_type_error::jsonio_type_error(const std::string& type):
    jsonio_error("Unsupported type: \"" + type + "\".")
{}

// Public functions - read and write directly from and to files

std::variant<arb::decor, arb::cable_cell_parameter_set> load_json(std::istream& s) {
    nlohmann::json json_data;
    try {
        s >> json_data;
    }
    catch (std::exception& e) {
        throw jsonio_json_parse_error(e.what());
    }
    if (!json_data.count("version")) {
        throw jsonio_missing_field("version");
    }
    if (auto version = json_data.at("version"); version != JSONIO_VERSION) {
        throw jsonio_version_error(version);
    }
    if (!json_data.count("type")) {
        throw jsonio_missing_field("type");
    }
    if (!json_data.count("data")) {
        throw jsonio_missing_field("data");
    }

    auto type = json_data.at("type");
    auto data = json_data.at("data");

    if (type == "global-parameters") {
        return data.get<arb::cable_cell_parameter_set>();
    } else if (type == "decor") {
        return data.get<arb::decor>();
    }
    throw jsonio_type_error(type);
}

void store_json(const arb::cable_cell_parameter_set& params, std::ostream& s) {
    nlohmann::json json_set;
    json_set["version"] = JSONIO_VERSION;
    json_set["type"] = "global-parameters";
    json_set["data"] = params;
    s << std::setw(2) << json_set;
};

void store_json(const arb::decor& decor, std::ostream& s) {
    nlohmann::json json_decor;
    json_decor["version"] = JSONIO_VERSION;
    json_decor["type"] = "decor";
    json_decor["data"] = decor;
    s << std::setw(2) << json_decor;
}

} // namespace arborio