#include <regex>
#include <string>
#include <unordered_set>

#include "expression.hpp"
#include "module.hpp"
#include "printerutil.hpp"

ARB_LIBMODCC_API std::vector<std::string> namespace_components(const std::string& ns) {
    static std::regex ns_regex("([^:]+)(?:::|$)");

    std::vector<std::string> components;
    auto i = std::sregex_iterator(ns.begin(), ns.end(), ns_regex);
    while (i!=std::sregex_iterator()) {
        components.push_back(i++->str(1));
    }

    return components;
}

ARB_LIBMODCC_API std::vector<LocalVariable*> indexed_locals(scope_ptr scope) {
    std::vector<LocalVariable*> vars;
    for (auto& entry: scope->locals()) {
        LocalVariable* local = entry.second->is_local_variable();
        if (local && local->is_indexed()) {
            vars.push_back(local);
        }
    }
    return vars;
}

ARB_LIBMODCC_API std::vector<LocalVariable*> pure_locals(scope_ptr scope) {
    std::vector<LocalVariable*> vars;
    for (auto& entry: scope->locals()) {
        LocalVariable* local = entry.second->is_local_variable();
        if (local && !local->is_arg() && !local->is_indexed()) {
            vars.push_back(local);
        }
    }
    return vars;
}

ARB_LIBMODCC_API std::vector<ProcedureExpression*> normal_procedures(const Module& m) {
    std::vector<ProcedureExpression*> procs;

    for (auto& sym: m.symbols()) {
        auto proc = sym.second->is_procedure();
        if (proc && proc->kind()==procedureKind::normal && !proc->is_api_method()
            && !proc->is_net_receive() && !proc->is_post_event()) {
            procs.push_back(proc);
        }
    }

    return procs;
}

ARB_LIBMODCC_API public_variable_ids_t public_variable_ids(const Module& m) {
    public_variable_ids_t ids;
    ids.state_ids = m.state_block().state_variables;

    std::unordered_set<std::string> range_varnames;
    for (const auto& sym: m.symbols()) {
        if (auto var = sym.second->is_variable()) {
            if (var->is_range()) {
                range_varnames.insert(var->name());
            }
        }
    }

    for (const Id& id: m.parameter_block().parameters) {
        if (range_varnames.count(id.token.spelling)) {
            ids.range_parameter_ids.push_back(id);
        }
        else {
            ids.global_parameter_ids.push_back(id);
        }
    }

    for (auto const & id : m.white_noise_block().parameters) {
        auto it = m.white_noise_block().used.find(id.name());
        if (it != m.white_noise_block().used.end()) {
            ids.white_noise_ids.push_back(std::make_pair(id, it->second));
        }
    }

    return ids;
}

ARB_LIBMODCC_API module_variables_t local_module_variables(const Module& m) {
    module_variables_t mv;

    for (auto& sym: m.symbols()) {
        auto v = sym.second->is_variable();
        if (v && v->linkage()==linkageKind::local) {
            (v->is_range()? mv.arrays: mv.scalars).push_back(v);
        }
    }

    return mv;
}

ARB_LIBMODCC_API std::vector<ProcedureExpression*> module_normal_procedures(const Module& m) {
    std::vector<ProcedureExpression*> procs;
    for (auto& sym: m.symbols()) {
        auto p = sym.second->is_procedure();
        if (p && p->kind()==procedureKind::normal) {
            procs.push_back(p);
        }
    }

    return procs;
}

ARB_LIBMODCC_API APIMethod* find_api_method(const Module& m, const char* which) {
    auto it = m.symbols().find(which);
    return  it==m.symbols().end()? nullptr: it->second->is_api_method();
}

ARB_LIBMODCC_API NetReceiveExpression* find_net_receive(const Module& m) {
    auto it = m.symbols().find("net_receive");
    return it==m.symbols().end()? nullptr: it->second->is_net_receive();
}

ARB_LIBMODCC_API PostEventExpression* find_post_event(const Module& m) {
    auto it = m.symbols().find("post_event");
    return it==m.symbols().end()? nullptr: it->second->is_post_event();
}

bool indexed_variable_info::scalar() const { return index_var_kind==index_kind::none; }

std::string indexed_variable_info::inner_index_var() const {
    if (index_var_kind == index_kind::cell) return node_index_var;
    return {};
}

std::string indexed_variable_info::outer_index_var() const {
    switch(index_var_kind) {
        case index_kind::node: return node_index_var;
        case index_kind::cell: return cell_index_var;
        case index_kind::other: return other_index_var;
        default: return {};
    }
}

ARB_LIBMODCC_API indexed_variable_info decode_indexed_variable(IndexedVariable* sym) {
    indexed_variable_info v;
    v.node_index_var = "node_index";
    v.index_var_kind = index_kind::node;
    v.scale = 1;
    v.accumulate = true;
    v.additive = false;
    v.readonly = true;
    v.always_use_weight = true;

    std::string ion_pfx;
    if (sym->is_ion()) {
        ion_pfx = "ion_"+sym->ion_channel();
        v.node_index_var = ion_pfx+"_index";
    }

    switch (sym->data_source()) {
    case sourceKind::voltage:
        v.data_var="vec_v";
        v.readonly = true;
        break;
    case sourceKind::peer_voltage:
        v.data_var="vec_v";
        v.other_index_var = "peer_index";
        v.node_index_var = "";
        v.index_var_kind = index_kind::other;
        v.readonly = true;
        break;
    case sourceKind::current_density:
        v.data_var = "vec_i";
        v.readonly = false;
        v.scale = 0.1;
        break;
    case sourceKind::current:
        // unit scale; sourceKind for point processes updating current variable.
        v.data_var = "vec_i";
        v.readonly = false;
        break;
    case sourceKind::conductivity:
        v.data_var = "vec_g";
        v.readonly = false;
        v.scale = 0.1;
        break;
    case sourceKind::conductance:
        // unit scale; sourceKind for point processes updating conductivity.
        v.data_var = "vec_g";
        v.readonly = false;
        break;
    case sourceKind::ion_current_density:
        v.data_var = ion_pfx+".current_density";
        v.scale = 0.1;
        v.readonly = false;
        break;
    case sourceKind::ion_current:
        // unit scale; sourceKind for point processes updating an ionic current variable.
        v.data_var = ion_pfx+".current_density";
        v.readonly = false;
        break;
    case sourceKind::ion_revpot:
        v.data_var = ion_pfx+".reversal_potential";
        v.accumulate = false;
        v.readonly = false;
        break;
    case sourceKind::ion_iconc:
        v.data_var = ion_pfx+".internal_concentration";
        v.readonly = false;
        v.always_use_weight = false;
        break;
    case sourceKind::ion_diffusive:
        v.data_var = ion_pfx+".diffusive_concentration";
        v.readonly = false;
        v.accumulate = false;
        v.additive = true;
        break;
    case sourceKind::ion_econc:
        v.data_var = ion_pfx+".external_concentration";
        v.readonly = false;
        v.always_use_weight = false;
        break;
    case sourceKind::ion_valence:
        v.data_var = ion_pfx+".ionic_charge";
        v.node_index_var = ""; // scalar global
        v.index_var_kind = index_kind::none;
        v.readonly = true;
        break;
    case sourceKind::temperature:
        v.data_var = "temperature_degC";
        v.readonly = true;
        break;
    case sourceKind::diameter:
        v.data_var = "diam_um";
        v.readonly = true;
        break;
    case sourceKind::area:
        v.data_var = "area_um2";
        v.readonly = true;
        break;
    default:
        throw compiler_exception(pprintf("unrecognized indexed data source: %", sym), sym->location());
    }

    return v;
}
