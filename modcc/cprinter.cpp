#include <algorithm>
#include <string>
#include <unordered_set>

#include "cexpr_emit.hpp"
#include "cprinter.hpp"
#include "lexer.hpp"

/******************************************************************************
                              CPrinter driver
******************************************************************************/

std::string CPrinter::emit_source() {
    // make a list of vector types, both parameters and assigned
    // and a list of all scalar types
    std::vector<VariableExpression*> scalar_variables;
    std::vector<VariableExpression*> array_variables;

    for(auto& sym: module_->symbols()) {
        if(auto var = sym.second->is_variable()) {
            if(var->is_range()) {
                array_variables.push_back(var);
            }
            else {
                scalar_variables.push_back(var);
            }
        }
    }

    std::string module_name = module_->module_name();

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    emit_headers();

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    std::string class_name = "mechanism_" + module_name;

    text_.add_line("namespace arb { namespace multicore {");
    text_.add_line();
    text_.add_line("template<class Backend>");
    text_.add_line("class " + class_name + " : public mechanism<Backend> {");
    text_.add_line("public:");
    text_.increase_indentation();
    text_.add_line("using base = mechanism<Backend>;");
    text_.add_line("using value_type  = typename base::value_type;");
    text_.add_line("using size_type   = typename base::size_type;");
    text_.add_line();
    text_.add_line("using array = typename base::array;");
    text_.add_line("using iarray  = typename base::iarray;");
    text_.add_line("using view   = typename base::view;");
    text_.add_line("using iview  = typename base::iview;");
    text_.add_line("using const_view = typename base::const_view;");
    text_.add_line("using const_iview = typename base::const_iview;");
    text_.add_line("using ion_type = typename base::ion_type;");
    text_.add_line("using deliverable_event_stream_state = typename base::deliverable_event_stream_state;");
    text_.add_line();

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    for(auto& ion: module_->neuron_block().ions) {
        auto tname = "Ion" + ion.name;
        text_.add_line("struct " + tname + " {");
        text_.increase_indentation();
        for(auto& field : ion.read) {
            text_.add_line("view " + field.spelling + ";");
        }
        for(auto& field : ion.write) {
            text_.add_line("view " + field.spelling + ";");
        }
        text_.add_line("iarray index;");
        text_.add_line("std::size_t memory() const { return sizeof(size_type)*index.size(); }");
        text_.add_line("std::size_t size() const { return index.size(); }");
        text_.decrease_indentation();
        text_.add_line("};");
        text_.add_line(tname + " ion_" + ion.name + ";");
    }

    //////////////////////////////////////////////
    // constructor
    //////////////////////////////////////////////
    int num_vars = array_variables.size();
    text_.add_line();
    text_.add_line(class_name + "(size_type mech_id, const_iview vec_ci, const_view vec_t, const_view vec_t_to, const_view vec_dt, view vec_v, view vec_i, array&& weights, iarray&& node_index)");
    text_.add_line(":   base(mech_id, vec_ci, vec_t, vec_t_to, vec_dt, vec_v, vec_i, std::move(node_index))");
    text_.add_line("{");
    text_.increase_indentation();
    text_.add_gutter() << "size_type num_fields = " << num_vars << ";";
    text_.end_line();

    text_.add_line();
    text_.add_line("// calculate the padding required to maintain proper alignment of sub arrays");
    text_.add_line("auto alignment  = data_.alignment();");
    text_.add_line("auto field_size_in_bytes = sizeof(value_type)*size();");
    text_.add_line("auto remainder  = field_size_in_bytes % alignment;");
    text_.add_line("auto padding    = remainder ? (alignment - remainder)/sizeof(value_type) : 0;");
    text_.add_line("auto field_size = size()+padding;");

    text_.add_line();
    text_.add_line("// allocate memory");
    text_.add_line("data_ = array(field_size*num_fields, std::numeric_limits<value_type>::quiet_NaN());");

    // assign the sub-arrays
    // replace this : data_(1*n, 2*n);
    //    with this : data_(1*field_size, 1*field_size+n);

    text_.add_line();
    text_.add_line("// asign the sub-arrays");
    for(int i=0; i<num_vars; ++i) {
        char namestr[128];
        sprintf(namestr, "%-15s", array_variables[i]->name().c_str());
        text_.add_gutter() << namestr << " = data_("
                           << i << "*field_size, " << i+1 << "*size());";
        text_.end_line();
    }
    text_.add_line();

    // copy in the weights
    text_.add_line("// add the user-supplied weights for converting from current density");
    text_.add_line("// to per-compartment current in nA");
    text_.add_line("if (weights.size()) {");
    text_.increase_indentation();
    text_.add_line("memory::copy(weights, weights_(0, size()));");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line("else {");
    text_.increase_indentation();
    text_.add_line("memory::fill(weights_, 1.0);");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    text_.add_line("// set initial values for variables and parameters");
    for(auto const& var : array_variables) {
        double val = var->value();
        // only non-NaN fields need to be initialized, because data_
        // is NaN by default
        std::string pointer_name = var->name()+".data()";
        if(val == val) {
            text_.add_gutter() << "std::fill(" << pointer_name << ", "
                                               << pointer_name << "+size(), "
                                               << val << ");";
            text_.end_line();
        }
    }

    text_.add_line();
    text_.decrease_indentation();
    text_.add_line("}");

    //////////////////////////////////////////////
    //////////////////////////////////////////////

    text_.add_line();
    text_.add_line("using base::size;");
    text_.add_line();

    text_.add_line("std::size_t memory() const override {");
    text_.increase_indentation();
    text_.add_line("auto s = std::size_t{0};");
    text_.add_line("s += data_.size()*sizeof(value_type);");
    for(auto& ion: module_->neuron_block().ions) {
        text_.add_line("s += ion_" + ion.name + ".memory();");
    }
    text_.add_line("return s;");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    text_.add_line("std::string name() const override {");
    text_.increase_indentation();
    text_.add_line("return \"" + module_name + "\";");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    std::string kind_str = module_->kind() == moduleKind::density
                            ? "mechanismKind::density"
                            : "mechanismKind::point";
    text_.add_line("mechanismKind kind() const override {");
    text_.increase_indentation();
    text_.add_line("return " + kind_str + ";");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    // Implement `set_weights` method.
    text_.add_line("void set_weights(array&& weights) override {");
    text_.increase_indentation();
    text_.add_line("memory::copy(weights, weights_(0, size()));");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    /***************************************************************************
     *
     *   ion channels have the following fields :
     *
     *       ---------------------------------------------------
     *       label   Ca      Na      K   name
     *       ---------------------------------------------------
     *       iX      ica     ina     ik  current
     *       eX      eca     ena     ek  reversal_potential
     *       Xi      cai     nai     ki  internal_concentration
     *       Xo      cao     nao     ko  external_concentration
     *       gX      gca     gna     gk  conductance
     *       ---------------------------------------------------
     *
     **************************************************************************/

    // ion_spec uses_ion(ionKind k) const override
    text_.add_line("typename base::ion_spec uses_ion(ionKind k) const override {");
    text_.increase_indentation();
    text_.add_line("bool uses = false;");
    text_.add_line("bool writes_ext = false;");
    text_.add_line("bool writes_int = false;");
    for (auto k: {ionKind::Na, ionKind::Ca, ionKind::K}) {
        if (module_->has_ion(k)) {
            auto ion = *module_->find_ion(k);
            text_.add_line("if (k==ionKind::" + ion.name + ") {");
            text_.increase_indentation();
            text_.add_line("uses = true;");
            if (ion.writes_concentration_int()) text_.add_line("writes_int = true;");
            if (ion.writes_concentration_ext()) text_.add_line("writes_ext = true;");
            text_.decrease_indentation();
            text_.add_line("}");
        }
    }
    text_.add_line("return {uses, writes_int, writes_ext};");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    // void set_ion(ionKind k, ion_type& i) override
    text_.add_line("void set_ion(ionKind k, ion_type& i, std::vector<size_type>const& index) override {");
    text_.increase_indentation();
    for (auto k: {ionKind::Na, ionKind::Ca, ionKind::K}) {
        if (module_->has_ion(k)) {
            auto ion = *module_->find_ion(k);
            text_.add_line("if (k==ionKind::" + ion.name + ") {");
            text_.increase_indentation();
            auto n = ion.name;
            auto pre = "ion_"+n;
            text_.add_line(pre+".index = memory::make_const_view(index);");
            if (ion.uses_current())
                text_.add_line(pre+".i"+n+" = i.current();");
            if (ion.uses_rev_potential())
                text_.add_line(pre+".e"+n+" = i.reversal_potential();");
            if (ion.uses_concentration_int())
                text_.add_line(pre+"."+n+"i = i.internal_concentration();");
            if (ion.uses_concentration_ext())
                text_.add_line(pre+"."+n+"o = i.external_concentration();");
            text_.add_line("return;");
            text_.decrease_indentation();
            text_.add_line("}");
        }
    }
    text_.add_line("throw std::domain_error(arb::util::pprintf(\"mechanism % does not support ion type\\n\", name()));");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    //////////////////////////////////////////////

    auto proctest = [] (procedureKind k) {
        return is_in(k, {procedureKind::normal, procedureKind::api, procedureKind::net_receive});
    };
    bool override_deliver_events = false;
    for(auto const& var: module_->symbols()) {
        auto isproc = var.second->kind()==symbolKind::procedure;
        if(isproc) {
            auto proc = var.second->is_procedure();
            if(proctest(proc->kind())) {
                proc->accept(this);
            }
            override_deliver_events |= proc->kind()==procedureKind::net_receive;
        }
    }

    if(override_deliver_events) {
        text_.add_line("void deliver_events(const deliverable_event_stream_state& events) override {");
        text_.increase_indentation();
        text_.add_line("auto ncell = events.n_streams();");
        text_.add_line("for (size_type c = 0; c<ncell; ++c) {");
        text_.increase_indentation();

        text_.add_line("auto begin = events.begin_marked(c);");
        text_.add_line("auto end = events.end_marked(c);");
        text_.add_line("for (auto p = begin; p<end; ++p) {");
        text_.increase_indentation();
        text_.add_line("if (p->mech_id==mech_id_) net_receive(p->mech_index, p->weight);");
        text_.decrease_indentation();
        text_.add_line("}");
        text_.decrease_indentation();
        text_.add_line("}");
        text_.decrease_indentation();
        text_.add_line("}");
        text_.add_line();
    }

    if(module_->write_backs().size()) {
        text_.add_line("void write_back() override {");
        text_.increase_indentation();

        text_.add_line("const size_type n_ = node_index_.size();");
        for (auto& w: module_->write_backs()) {
            auto& src = w.source_name;
            auto tgt = w.target_name;
            tgt.erase(tgt.begin(), tgt.begin()+tgt.find('_')+1);
            auto istore = ion_store(w.ion_kind)+".";

            text_.add_line();
            text_.add_line("auto "+src+"_out_ = util::indirect_view("+istore+tgt+", "+istore+"index);");
            text_.add_line("for (size_type i_ = 0; i_ < n_; ++i_) {");
            text_.increase_indentation();
            text_.add_line("// 1/10 magic number due to unit normalisation");
            text_.add_line(src+"_out_["+istore+"index[i_]] += value_type(0.1)*weights_[i_]*"+src+"[i_];");
            text_.decrease_indentation(); text_.add_line("}");
            
        }
        text_.decrease_indentation(); text_.add_line("}");
    }
    text_.add_line();

    // TODO: replace field_info() generation implemenation with separate schema info generation
    // as per #349.
    auto field_info_string = [](const std::string& kind, const Id& id) {
        return  "field_spec{field_spec::" + kind + ", " +
                "\"" + id.unit_string() + "\", " +
                (id.has_value()? id.value: "0") +
                (id.has_range()? ", " + id.range.first.spelling + "," + id.range.second.spelling: "") +
                "}";
    };

    std::unordered_set<std::string> scalar_set;
    for (auto& v: scalar_variables) {
        scalar_set.insert(v->name());
    }

    std::vector<Id> global_param_ids;
    std::vector<Id> instance_param_ids;

    for (const Id& id: module_->parameter_block().parameters) {
        auto var = id.token.spelling;
        (scalar_set.count(var)? global_param_ids: instance_param_ids).push_back(id);
    }
    const std::vector<Id>& state_ids = module_->state_block().state_variables;

    text_.add_line("util::optional<field_spec> field_info(const char* id) const /* override */ {");
    text_.increase_indentation();
    text_.add_line("static const std::pair<const char*, field_spec> field_tbl[] = {");
    text_.increase_indentation();
    for (const auto& id: global_param_ids) {
        auto var = id.token.spelling;
        text_.add_line("{\""+var+"\", "+field_info_string("global",id )+"},");
    }
    for (const auto& id: instance_param_ids) {
        auto var = id.token.spelling;
        text_.add_line("{\""+var+"\", "+field_info_string("parameter", id)+"},");
    }
    for (const auto& id: state_ids) {
        auto var = id.token.spelling;
        text_.add_line("{\""+var+"\", "+field_info_string("state", id)+"},");
    }
    text_.decrease_indentation();
    text_.add_line("};");
    text_.add_line();
    text_.add_line("auto* info = util::table_lookup(field_tbl, id);");
    text_.add_line("return info? util::just(*info): util::nothing;");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    if (!instance_param_ids.empty() || !state_ids.empty()) {
        text_.add_line("view base::* field_view_ptr(const char* id) const override {");
        text_.increase_indentation();
        text_.add_line("static const std::pair<const char*, view "+class_name+"::*> field_tbl[] = {");
        text_.increase_indentation();
        for (const auto& id: instance_param_ids) {
            auto var = id.token.spelling;
            text_.add_line("{\""+var+"\", &"+class_name+"::"+var+"},");
        }
        for (const auto& id: state_ids) {
            auto var = id.token.spelling;
            text_.add_line("{\""+var+"\", &"+class_name+"::"+var+"},");
        }
        text_.decrease_indentation();
        text_.add_line("};");
        text_.add_line();
        text_.add_line("auto* pptr = util::table_lookup(field_tbl, id);");
        text_.add_line("return pptr? static_cast<view base::*>(*pptr): nullptr;");
        text_.decrease_indentation();
        text_.add_line("}");
        text_.add_line();
    }

    if (!global_param_ids.empty()) {
        text_.add_line("value_type base::* field_value_ptr(const char* id) const override {");
        text_.increase_indentation();
        text_.add_line("static const std::pair<const char*, value_type "+class_name+"::*> field_tbl[] = {");
        text_.increase_indentation();
        for (const auto& id: global_param_ids) {
            auto var = id.token.spelling;
            text_.add_line("{\""+var+"\", &"+class_name+"::"+var+"},");
        }
        text_.decrease_indentation();
        text_.add_line("};");
        text_.add_line();
        text_.add_line("auto* pptr = util::table_lookup(field_tbl, id);");
        text_.add_line("return pptr? static_cast<value_type base::*>(*pptr): nullptr;");
        text_.decrease_indentation();
        text_.add_line("}");
        text_.add_line();
    }

    //////////////////////////////////////////////
    //////////////////////////////////////////////

    text_.add_line("array data_;");
    for(auto var: array_variables) {
        text_.add_line("view " + var->name() + ";");
    }

    for(auto var: scalar_variables) {
        double val = var->value();
        // test the default value for NaN
        // useful for error propogation from bad initial conditions
        if(val==val) {
            text_.add_gutter() << "value_type " << var->name() << " = " << val << ";";
            text_.end_line();
        }
        else {
            text_.add_line("value_type " + var->name() + " = 0;");
        }
    }

    text_.add_line();
    text_.add_line("using base::mech_id_;");
    text_.add_line("using base::vec_ci_;");
    text_.add_line("using base::vec_t_;");
    text_.add_line("using base::vec_t_to_;");
    text_.add_line("using base::vec_dt_;");
    text_.add_line("using base::vec_v_;");
    text_.add_line("using base::vec_i_;");
    text_.add_line("using base::node_index_;");

    text_.add_line();
    text_.decrease_indentation();
    text_.add_line("};");
    text_.add_line();

    text_.add_line("}} // namespaces");
    return text_.str();
}



void CPrinter::emit_headers() {
    text_.add_line("#pragma once");
    text_.add_line();
    text_.add_line("#include <cmath>");
    text_.add_line("#include <limits>");
    text_.add_line();
    text_.add_line("#include <mechanism.hpp>");
    text_.add_line("#include <algorithms.hpp>");
    text_.add_line("#include <backends/event.hpp>");
    text_.add_line("#include <backends/multi_event_stream_state.hpp>");
    text_.add_line("#include <util/pprintf.hpp>");
    text_.add_line("#include <util/simple_table.hpp>");
    text_.add_line();
}

/******************************************************************************
                              CPrinter
******************************************************************************/

void CPrinter::visit(Expression *e) {
    throw compiler_exception(
        "CPrinter doesn't know how to print " + e->to_string(),
        e->location());
}

void CPrinter::visit(LocalDeclaration *e) {
}

void CPrinter::visit(Symbol *e) {
    throw compiler_exception("I don't know how to print raw Symbol " + e->to_string(),
                             e->location());
}

void CPrinter::visit(LocalVariable *e) {
    std::string const& name = e->name();
    text_ << name;
    if(is_ghost_local(e)) {
        text_ << "[j_]";
    }
}

void CPrinter::visit(NumberExpression *e) {
    cexpr_emit(e, text_.text(), this);
}

void CPrinter::visit(IdentifierExpression *e) {
    e->symbol()->accept(this);
}

void CPrinter::visit(VariableExpression *e) {
    text_ << e->name();
    if(e->is_range()) {
        text_ << "[i_]";
    }
}

void CPrinter::visit(IndexedVariable *e) {
    text_ << e->index_name() << "[i_]";
}

void CPrinter::visit(CellIndexedVariable *e) {
    text_ << e->index_name() << "[i_]";
}

void CPrinter::visit(UnaryExpression *e) {
    cexpr_emit(e, text_.text(), this);
}

void CPrinter::visit(BlockExpression *e) {
    // ------------- declare local variables ------------- //
    // only if this is the outer block
    if(!e->is_nested()) {
        std::vector<std::string> names;
        for(auto& symbol : e->scope()->locals()) {
            auto sym = symbol.second.get();
            // input variables are declared earlier, before the
            // block body is printed
            if(is_stack_local(sym) && !is_input(sym)) {
                names.push_back(sym->name());
            }
        }
        if(names.size()>0) {
            text_.add_gutter() << "value_type " << *(names.begin());
            for(auto it=names.begin()+1; it!=names.end(); ++it) {
                text_ << ", " << *it;
            }
            text_.end_line(";");
        }
    }

    // ------------- statements ------------- //
    for(auto& stmt : e->statements()) {
        if(stmt->is_local_declaration()) continue;

        // these all must be handled
        text_.add_gutter();
        stmt->accept(this);
        if (not stmt->is_if()) {
            text_.end_line(";");
        }
    }
}

void CPrinter::visit(IfExpression *e) {
    // for now we remove the brackets around the condition because
    // the binary expression printer adds them, and we want to work
    // around the -Wparentheses-equality warning
    text_ << "if(";
    e->condition()->accept(this);
    text_ << ") {\n";
    increase_indentation();
    e->true_branch()->accept(this);
    decrease_indentation();
    text_.add_line("}");
    // check if there is a false-branch, i.e. if
    // there is an "else" branch to print
    if (auto fb = e->false_branch()) {
        text_.add_gutter() << "else ";
        // use recursion for "else if"
        if (fb->is_if()) {
            fb->accept(this);
        }
        // otherwise print the "else" block
        else {
            text_ << "{\n";
            increase_indentation();
            fb->accept(this);
            decrease_indentation();
            text_.add_line("}");
        }
    }
}

// NOTE: net_receive() is classified as a ProcedureExpression
void CPrinter::visit(ProcedureExpression *e) {
    // print prototype
    text_.add_gutter() << "void " << e->name() << "(int i_";
    for(auto& arg : e->args()) {
        text_ << ", value_type " << arg->is_argument()->name();
    }
    if(e->kind() == procedureKind::net_receive) {
        text_.end_line(") override {");
    }
    else {
        text_.end_line(") {");
    }

    if(!e->scope()) { // error: semantic analysis has not been performed
        throw compiler_exception(
            "CPrinter attempt to print Procedure " + e->name()
            + " for which semantic analysis has not been performed",
            e->location());
    }

    // print body
    increase_indentation();
    e->body()->accept(this);

    // close the function body
    decrease_indentation();
    text_.add_line("}");
    text_.add_line();
}

void CPrinter::visit(APIMethod *e) {
    // print prototype
    text_.add_gutter() << "void " << e->name() << "() override {";
    text_.end_line();

    if(!e->scope()) { // error: semantic analysis has not been performed
        throw compiler_exception(
            "CPrinter attempt to print APIMethod " + e->name()
            + " for which semantic analysis has not been performed",
            e->location());
    }

    // only print the body if it has contents
    if(e->is_api_method()->body()->statements().size()) {
        increase_indentation();

        // create local indexed views
        for(auto &symbol : e->scope()->locals()) {
            auto var = symbol.second->is_local_variable();
            if (!var->is_indexed()) continue;

            auto external = var->external_variable();
            auto const& name = var->name();
            auto const& index_name = external->index_name();

            text_.add_gutter();
            text_ << "auto " + index_name + " = ";

            if(external->is_cell_indexed_variable()) {
                text_ << "util::indirect_view(util::indirect_view(" + index_name + "_, vec_ci_), node_index_);\n";
            }
            else if(external->is_ion()) {
                auto channel = external->ion_channel();
                auto iname = ion_store(channel);
                text_ << "util::indirect_view(" << iname << "." << name << ", " << ion_store(channel) << ".index);\n";
            }
            else {
                text_ << "util::indirect_view(" + index_name + "_, node_index_);\n";
            }
        }

        // get loop dimensions
        text_.add_line("int n_ = node_index_.size();");

        print_APIMethod(e);
    }

    // close up the loop body
    text_.add_line("}");
    text_.add_line();
}

void CPrinter::emit_api_loop(APIMethod* e,
                             const std::string& start,
                             const std::string& end,
                             const std::string& inc) {
    text_.add_gutter();
    text_ << "for (" << start << "; " << end << "; " << inc << ") {";
    text_.end_line();
    text_.increase_indentation();

    // loads from external indexed arrays
    for(auto &symbol : e->scope()->locals()) {
        auto var = symbol.second->is_local_variable();
        if(is_input(var)) {
            auto ext = var->external_variable();
            text_.add_gutter() << "value_type ";
            var->accept(this);
            text_ << " = ";
            ext->accept(this);
            text_.end_line(";");
        }
    }

    // print the body of the loop
    e->body()->accept(this);

    // perform update of external variables (currents etc)
    for(auto &symbol : e->scope()->locals()) {
        auto var = symbol.second->is_local_variable();
        if(is_output(var)) {
            auto ext = var->external_variable();
            text_.add_gutter();
            ext->accept(this);
            text_ << (ext->op() == tok::plus ? " += " : " -= ");
            var->accept(this);
            text_.end_line(";");
        }
    }

    text_.decrease_indentation();
    text_.add_line("}");
}

void CPrinter::print_APIMethod(APIMethod* e) {
    emit_api_loop(e, "int i_ = 0", "i_ < n_", "++i_");
    decrease_indentation();

    return;
}

void CPrinter::visit(CallExpression *e) {
    text_ << e->name() << "(i_";
    for(auto& arg: e->args()) {
        text_ << ", ";
        arg->accept(this);
    }
    text_ << ")";
}

void CPrinter::visit(BinaryExpression *e) {
    cexpr_emit(e, text_.text(), this);
}

