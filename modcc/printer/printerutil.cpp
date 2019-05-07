#include <regex>
#include <string>
#include <unordered_set>

#include "expression.hpp"
#include "module.hpp"
#include "printerutil.hpp"

std::vector<std::string> namespace_components(const std::string& ns) {
    static std::regex ns_regex("([^:]+)(?:::|$)");

    std::vector<std::string> components;
    auto i = std::sregex_iterator(ns.begin(), ns.end(), ns_regex);
    while (i!=std::sregex_iterator()) {
        components.push_back(i++->str(1));
    }

    return components;
}

std::vector<LocalVariable*> indexed_locals(scope_ptr scope) {
    std::vector<LocalVariable*> vars;
    for (auto& entry: scope->locals()) {
        LocalVariable* local = entry.second->is_local_variable();
        if (local && local->is_indexed()) {
            vars.push_back(local);
        }
    }
    return vars;
}

std::vector<LocalVariable*> pure_locals(scope_ptr scope) {
    std::vector<LocalVariable*> vars;
    for (auto& entry: scope->locals()) {
        LocalVariable* local = entry.second->is_local_variable();
        if (local && !local->is_arg() && !local->is_indexed()) {
            vars.push_back(local);
        }
    }
    return vars;
}

std::vector<ProcedureExpression*> normal_procedures(const Module& m) {
    std::vector<ProcedureExpression*> procs;

    for (auto& sym: m.symbols()) {
        auto proc = sym.second->is_procedure();
        if (proc && proc->kind()==procedureKind::normal && !proc->is_api_method() && !proc->is_net_receive()) {
            procs.push_back(proc);
        }
    }

    return procs;
}

public_variable_ids_t public_variable_ids(const Module& m) {
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

    return ids;
}

module_variables_t local_module_variables(const Module& m) {
    module_variables_t mv;

    for (auto& sym: m.symbols()) {
        auto v = sym.second->is_variable();
        if (v && v->linkage()==linkageKind::local) {
            (v->is_range()? mv.arrays: mv.scalars).push_back(v);
        }
    }

    return mv;
}

std::vector<ProcedureExpression*> module_normal_procedures(const Module& m) {
    std::vector<ProcedureExpression*> procs;
    for (auto& sym: m.symbols()) {
        auto p = sym.second->is_procedure();
        if (p && p->kind()==procedureKind::normal) {
            procs.push_back(p);
        }
    }

    return procs;
}

APIMethod* find_api_method(const Module& m, const char* which) {
    auto it = m.symbols().find(which);
    return  it==m.symbols().end()? nullptr: it->second->is_api_method();
}

NetReceiveExpression* find_net_receive(const Module& m) {
    auto it = m.symbols().find("net_receive");
    return it==m.symbols().end()? nullptr: it->second->is_net_receive();
}

indexed_variable_info decode_indexed_variable(IndexedVariable* sym) {
    std::string data_var, ion_pfx;
    std::string index_var = "node_index_";

    if (sym->is_ion()) {
        ion_pfx = "ion_"+to_string(sym->ion_channel())+"_";
        index_var = ion_pfx+"index_";
    }

    switch (sym->data_source()) {
    case sourceKind::voltage:
        data_var="vec_v_";
        break;
    case sourceKind::current:
        data_var="vec_i_";
        break;
    case sourceKind::conductivity:
        data_var="vec_g_";
        break;
    case sourceKind::dt:
        data_var="vec_dt_";
        break;
    case sourceKind::ion_current:
        data_var=ion_pfx+".current_density";
        break;
    case sourceKind::ion_revpot:
        data_var=ion_pfx+".reversal_potential";
        break;
    case sourceKind::ion_iconc:
        data_var=ion_pfx+".internal_concentration";
        break;
    case sourceKind::ion_econc:
        data_var=ion_pfx+".external_concentration";
        break;
    case sourceKind::temperature:
        data_var="temperature_degC_";
        index_var=""; // scalar global
        break;
    default:
        throw compiler_exception(pprintf("unrecognized indexed data source: %", sym), sym->location());
    }

    return {data_var, index_var};
}
