#include <algorithm>

#include "cprinter.hpp"
#include "lexer.hpp"

/******************************************************************************
                              CPrinter driver
******************************************************************************/

CPrinter::CPrinter(Module &m, bool o)
:   module_(&m),
    optimize_(o)
{
    // make a list of vector types, both parameters and assigned
    // and a list of all scalar types
    std::vector<VariableExpression*> scalar_variables;
    std::vector<VariableExpression*> array_variables;
    for(auto& sym: m.symbols()) {
        if(auto var = sym.second->is_variable()) {
            if(var->is_range()) {
                array_variables.push_back(var);
            }
            else {
                scalar_variables.push_back(var);
            }
        }
    }

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    text_.add_line("#pragma once");
    text_.add_line();
    text_.add_line("#include <cmath>");
    text_.add_line("#include <limits>");
    text_.add_line();
    text_.add_line("#include <mechanism.hpp>");
    text_.add_line("#include <mechanism_interface.hpp>");
    text_.add_line("#include <algorithms.hpp>");
    text_.add_line();

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    std::string class_name = "mechanism_" + m.name();

    text_.add_line("namespace nest{ namespace mc{ namespace mechanisms{ namespace " + m.name() + "{");
    text_.add_line();
    text_.add_line("template<typename T, typename I>");
    text_.add_line("class " + class_name + " : public mechanism<T, I> {");
    text_.add_line("public:");
    text_.increase_indentation();
    text_.add_line("using base = mechanism<T, I>;");
    text_.add_line("using value_type  = typename base::value_type;");
    text_.add_line("using size_type   = typename base::size_type;");
    text_.add_line("using vector_type = typename base::vector_type;");
    text_.add_line("using view_type   = typename base::view_type;");
    text_.add_line("using index_type  = typename base::index_type;");
    text_.add_line("using index_view  = typename base::index_view;");
    text_.add_line("using const_index_view  = typename base::const_index_view;");
    text_.add_line("using indexed_view_type= typename base::indexed_view_type;");
    text_.add_line("using ion_type = typename base::ion_type;");
    text_.add_line();

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    for(auto& ion: m.neuron_block().ions) {
        auto tname = "Ion" + ion.name;
        text_.add_line("struct " + tname + " {");
        text_.increase_indentation();
        for(auto& field : ion.read) {
            text_.add_line("view_type " + field.spelling + ";");
        }
        for(auto& field : ion.write) {
            text_.add_line("view_type " + field.spelling + ";");
        }
        text_.add_line("index_type index;");
        text_.add_line("std::size_t memory() const { return sizeof(size_type)*index.size(); }");
        text_.add_line("std::size_t size() const { return index.size(); }");
        text_.decrease_indentation();
        text_.add_line("};");
        text_.add_line(tname + " ion_" + ion.name + ";");
    }
    text_.add_line();

    //////////////////////////////////////////////
    // constructor
    //////////////////////////////////////////////
    int num_vars = array_variables.size();
    text_.add_line(class_name + "(view_type vec_v, view_type vec_i, const_index_view node_index)");
    text_.add_line(":   base(vec_v, vec_i, node_index)");
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
    text_.add_line("data_ = vector_type(field_size * num_fields);");
    text_.add_line("data_(memory::all) = std::numeric_limits<value_type>::quiet_NaN();");

    // assign the sub-arrays
    // replace this : data_(1*n, 2*n);
    //    with this : data_(1*field_size, 1*field_size+n);

    text_.add_line();
    text_.add_line("// asign the sub-arrays");
    for(int i=0; i<num_vars; ++i) {
        char namestr[128];
        sprintf(namestr, "%-15s", array_variables[i]->name().c_str());
        if(optimize_) {
            text_.add_gutter() << namestr << " = data_.data() + "
                               << i << "*field_size;";
        }
        else {
            text_.add_gutter() << namestr << " = data_("
                               << i << "*field_size, " << i+1 << "*size());";
        }
        text_.end_line();
    }

    text_.add_line();
    text_.add_line("// set initial values for variables and parameters");
    for(auto const& var : array_variables) {
        double val = var->value();
        // only non-NaN fields need to be initialized, because data_
        // is NaN by default
        std::string pointer_name = var->name();
        if(!optimize_) pointer_name += ".data()";
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
    for(auto& ion: m.neuron_block().ions) {
        text_.add_line("s += ion_" + ion.name + ".memory();");
    }
    text_.add_line("return s;");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    text_.add_line("void set_params(value_type t_, value_type dt_) override {");
    text_.increase_indentation();
    text_.add_line("t = t_;");
    text_.add_line("dt = dt_;");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    text_.add_line("std::string name() const override {");
    text_.increase_indentation();
    text_.add_line("return \"" + m.name() + "\";");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    std::string kind_str = m.kind() == moduleKind::density
                            ? "mechanismKind::density"
                            : "mechanismKind::point";
    text_.add_line("mechanismKind kind() const override {");
    text_.increase_indentation();
    text_.add_line("return " + kind_str + ";");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    // return true/false indicating if cell has dependency on k
    auto const& ions = m.neuron_block().ions;
    auto find_ion = [&ions] (ionKind k) {
        return std::find_if(
            ions.begin(), ions.end(),
            [k](IonDep const& d) {return d.kind()==k;}
        );
    };
    auto has_ion = [&ions, find_ion] (ionKind k) {
        return find_ion(k) != ions.end();
    };

    // bool uses_ion(ionKind k) const override
    text_.add_line("bool uses_ion(ionKind k) const override {");
    text_.increase_indentation();
    text_.add_line("switch(k) {");
    text_.increase_indentation();
    text_.add_gutter()
        << "case ionKind::na : return "
        << (has_ion(ionKind::Na) ? "true" : "false") << ";";
    text_.end_line();
    text_.add_gutter()
        << "case ionKind::ca : return "
        << (has_ion(ionKind::Ca) ? "true" : "false") << ";";
    text_.end_line();
    text_.add_gutter()
        << "case ionKind::k  : return "
        << (has_ion(ionKind::K) ? "true" : "false") << ";";
    text_.end_line();
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line("return false;");
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

    // void set_ion(ionKind k, ion_type& i) override
    //      TODO: this is done manually, which isn't going to scale
    auto has_variable = [] (IonDep const& ion, std::string const& name) {
        if( std::find_if(ion.read.begin(), ion.read.end(),
                      [&name] (Token const& t) {return t.spelling==name;}
            ) != ion.read.end()
        ) return true;
        if( std::find_if(ion.write.begin(), ion.write.end(),
                      [&name] (Token const& t) {return t.spelling==name;}
            ) != ion.write.end()
        ) return true;
        return false;
    };
    text_.add_line("void set_ion(ionKind k, ion_type& i) override {");
    text_.increase_indentation();
    text_.add_line("using nest::mc::algorithms::index_into;");
    if(has_ion(ionKind::Na)) {
        auto ion = find_ion(ionKind::Na);
        text_.add_line("if(k==ionKind::na) {");
        text_.increase_indentation();
        text_.add_line("ion_na.index = index_into(i.node_index(), node_index_);");
        if(has_variable(*ion, "ina")) text_.add_line("ion_na.ina = i.current();");
        if(has_variable(*ion, "ena")) text_.add_line("ion_na.ena = i.reversal_potential();");
        if(has_variable(*ion, "nai")) text_.add_line("ion_na.nai = i.internal_concentration();");
        if(has_variable(*ion, "nao")) text_.add_line("ion_na.nao = i.external_concentration();");
        text_.add_line("return;");
        text_.decrease_indentation();
        text_.add_line("}");
    }
    if(has_ion(ionKind::Ca)) {
        auto ion = find_ion(ionKind::Ca);
        text_.add_line("if(k==ionKind::ca) {");
        text_.increase_indentation();
        text_.add_line("ion_ca.index = index_into(i.node_index(), node_index_);");
        if(has_variable(*ion, "ica")) text_.add_line("ion_ca.ica = i.current();");
        if(has_variable(*ion, "eca")) text_.add_line("ion_ca.eca = i.reversal_potential();");
        if(has_variable(*ion, "cai")) text_.add_line("ion_ca.cai = i.internal_concentration();");
        if(has_variable(*ion, "cao")) text_.add_line("ion_ca.cao = i.external_concentration();");
        text_.add_line("return;");
        text_.decrease_indentation();
        text_.add_line("}");
    }
    if(has_ion(ionKind::K)) {
        auto ion = find_ion(ionKind::K);
        text_.add_line("if(k==ionKind::k) {");
        text_.increase_indentation();
        text_.add_line("ion_k.index = index_into(i.node_index(), node_index_);");
        if(has_variable(*ion, "ik")) text_.add_line("ion_k.ik = i.current();");
        if(has_variable(*ion, "ek")) text_.add_line("ion_k.ek = i.reversal_potential();");
        if(has_variable(*ion, "ki")) text_.add_line("ion_k.ki = i.internal_concentration();");
        if(has_variable(*ion, "ko")) text_.add_line("ion_k.ko = i.external_concentration();");
        text_.add_line("return;");
        text_.decrease_indentation();
        text_.add_line("}");
    }
    text_.add_line("throw std::domain_error(nest::mc::util::pprintf(\"mechanism % does not support ion type\\n\", name()));");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    //////////////////////////////////////////////
    //////////////////////////////////////////////

    auto proctest = [] (procedureKind k) {
        return
            k == procedureKind::normal
                 || k == procedureKind::api
                 || k == procedureKind::net_receive;
    };
    for(auto &var : m.symbols()) {
        auto isproc = var.second->kind()==symbolKind::procedure;
        if(isproc )
        {
            auto proc = var.second->is_procedure();
            if(proctest(proc->kind())) {
                proc->accept(this);
            }
        }
    }

    //////////////////////////////////////////////
    //////////////////////////////////////////////

    text_.add_line("vector_type data_;");
    for(auto var: array_variables) {
        if(optimize_) {
            text_.add_line(
                "__declspec(align(vector_type::alignment())) value_type *"
                + var->name() + ";");
        }
        else {
            text_.add_line("view_type " + var->name() + ";");
        }
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
    text_.add_line("using base::vec_v_;");
    text_.add_line("using base::vec_i_;");
    text_.add_line("using base::vec_area_;");
    text_.add_line("using base::node_index_;");

    text_.add_line();
    //text_.add_line("DATA_PROFILE");
    text_.decrease_indentation();
    text_.add_line("};");
    text_.add_line();

    // print the helper type that provides the bridge from the mechanism to
    // the calling code
    text_.add_line("template<typename T, typename I>");
    text_.add_line("struct helper : public mechanism_helper<T, I> {");
    text_.increase_indentation();
    text_.add_line("using base = mechanism_helper<T, I>;");
    text_.add_line("using index_view  = typename base::index_view;");
    text_.add_line("using view_type  = typename base::view_type;");
    text_.add_line("using mechanism_ptr_type  = typename base::mechanism_ptr_type;");
    text_.add_gutter() << "using mechanism_type = " << class_name << "<T, I>;";
    text_.add_line();
    text_.add_line();

    text_.add_line("std::string");
    text_.add_line("name() const override");
    text_.add_line("{");
    text_.increase_indentation();
    text_.add_gutter() << "return \"" << m.name() << "\";";
    text_.add_line();
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    text_.add_line("mechanism_ptr<T,I>");
    text_.add_line("new_mechanism(view_type vec_v, view_type vec_i, index_view node_index) const override");
    text_.add_line("{");
    text_.increase_indentation();
    text_.add_line("return nest::mc::mechanisms::make_mechanism<mechanism_type>(vec_v, vec_i, node_index);");
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    text_.add_line("void");
    text_.add_line("set_parameters(mechanism_ptr_type&, parameter_list const&) const override");
    text_.add_line("{");
    text_.increase_indentation();
    // TODO : interface that writes parameter_list paramaters into the mechanism's storage
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    text_.decrease_indentation();
    text_.add_line("};");
    text_.add_line();

    text_.add_line("}}}} // namespaces");
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
    text_ << " " << e->value();
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

void CPrinter::visit(UnaryExpression *e) {
    auto b = (e->expression()->is_binary()!=nullptr);
    switch(e->op()) {
        case tok::minus :
            // place a space in front of minus sign to avoid invalid
            // expressions of the form : (v[i]--67)
            if(b) text_ << " -(";
            else  text_ << " -";
            e->expression()->accept(this);
            if(b) text_ << ")";
            return;
        case tok::exp :
            text_ << "exp(";
            e->expression()->accept(this);
            text_ << ")";
            return;
        case tok::cos :
            text_ << "cos(";
            e->expression()->accept(this);
            text_ << ")";
            return;
        case tok::sin :
            text_ << "sin(";
            e->expression()->accept(this);
            text_ << ")";
            return;
        case tok::log :
            text_ << "log(";
            e->expression()->accept(this);
            text_ << ")";
            return;
        default :
            throw compiler_exception(
                "CPrinter unsupported unary operator " + yellow(token_string(e->op())),
                e->location());
    }
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
            //for(auto it=names.begin(); it!=names.end(); ++it) {
            //    text_.add_gutter() << "value_type " << *it;
            //    text_.end_line("{0};");
            //}
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
        text_.end_line(";");
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
    text_.add_gutter();
    text_ << "}";
}

void CPrinter::visit(ProcedureExpression *e) {
    // ------------- print prototype ------------- //
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

    increase_indentation();

    e->body()->accept(this);

    // ------------- close up ------------- //
    decrease_indentation();
    text_.add_line("}");
    text_.add_line();
    return;
}

void CPrinter::visit(APIMethod *e) {
    // ------------- print prototype ------------- //
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
            if(var->is_indexed()) {
                auto const& name = var->name();
                auto const& index_name = var->external_variable()->index_name();
                text_.add_gutter();
                if(var->is_read()) text_ << "const ";
                text_ << "indexed_view_type " + index_name;
                auto channel = var->external_variable()->ion_channel();
                if(channel==ionKind::none) {
                    text_ << "(" + index_name + "_, node_index_);\n";
                }
                else {
                    auto iname = ion_store(channel);
                    text_ << "(" << iname << "." << name << ", "
                          << ion_store(channel) << ".index);\n";
                }
            }
        }

        // ------------- get loop dimensions ------------- //
        text_.add_line("int n_ = node_index_.size();");

        // hand off printing of loops to optimized or unoptimized backend
        if(optimize_) {
            print_APIMethod_optimized(e);
        }
        else {
            print_APIMethod_unoptimized(e);
        }
    }

    // ------------- close up ------------- //
    text_.add_line("}");
    text_.add_line();
}

void CPrinter::print_APIMethod_unoptimized(APIMethod* e) {
    //text_.add_line("START_PROFILE");

    // there can not be more than 1 instance of a density channel per grid point,
    // so we can assert that aliasing will not occur.
    if(optimize_) text_.add_line("#pragma ivdep");

    text_.add_line("for(int i_=0; i_<n_; ++i_) {");
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

    //text_.add_line("STOP_PROFILE");
    decrease_indentation();

    return;
}

void CPrinter::print_APIMethod_optimized(APIMethod* e) {
    // ------------- get mechanism properties ------------- //

    // make a list of all the local variables that have to be
    // written out to global memory via an index
    auto is_aliased = [this] (Symbol* s) -> LocalVariable* {
        if(is_output(s)) {
            return s->is_local_variable();
        }
        return nullptr;
    };

    std::vector<LocalVariable*> aliased_variables;
    if(is_point_process()) {
        for(auto &l : e->scope()->locals()) {
            if(auto var = is_aliased(l.second.get())) {
                aliased_variables.push_back(var);
            }
        }
    }
    aliased_output_ = aliased_variables.size()>0;

    // only proceed with optimized output if the ouputs are aliased
    // because all optimizations are for using ghost buffers to avoid
    // race conditions in vectorized code
    if(!aliased_output_) {
        print_APIMethod_unoptimized(e);
        return;
    }

    // ------------- block loop ------------- //

    text_.add_line("constexpr int BSIZE = 4;");
    text_.add_line("int NB = n_/BSIZE;");
    for(auto out: aliased_variables) {
        text_.add_line(
            "__declspec(align(vector_type::alignment())) value_type "
            + out->name() +  "[BSIZE];");
    }
    //text_.add_line("START_PROFILE");

    text_.add_line("for(int b_=0; b_<NB; ++b_) {");
    text_.increase_indentation();
    text_.add_line("int BSTART = BSIZE*b_;");
    text_.add_line("int i_ = BSTART;");


    // assert that memory accesses are not aliased because we will
    // use ghost arrays to ensure that write-back of point processes does
    // not lead to race conditions
    text_.add_line("#pragma ivdep");
    text_.add_line("for(int j_=0; j_<BSIZE; ++j_, ++i_) {");
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

    e->body()->accept(this);

    text_.decrease_indentation();
    text_.add_line("}"); // end inner compute loop

    text_.add_line("i_ = BSTART;");
    text_.add_line("for(int j_=0; j_<BSIZE; ++j_, ++i_) {");
    text_.increase_indentation();

    for(auto out: aliased_variables) {
        text_.add_gutter();
        auto ext = out->external_variable();
        ext->accept(this);
        text_ << (ext->op() == tok::plus ? " += " : " -= ");
        out->accept(this);
        text_.end_line(";");
    }

    text_.decrease_indentation();
    text_.add_line("}"); // end inner write loop
    text_.decrease_indentation();
    text_.add_line("}"); // end outer block loop

    // ------------- block tail loop ------------- //

    text_.add_line("int j_ = 0;");
    text_.add_line("#pragma ivdep");
    text_.add_line("for(int i_=NB*BSIZE; i_<n_; ++j_, ++i_) {");
    text_.increase_indentation();

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

    e->body()->accept(this);

    text_.decrease_indentation();
    text_.add_line("}"); // end inner compute loop
    text_.add_line("j_ = 0;");
    text_.add_line("for(int i_=NB*BSIZE; i_<n_; ++j_, ++i_) {");
    text_.increase_indentation();

    for(auto out: aliased_variables) {
        text_.add_gutter();
        auto ext = out->external_variable();
        ext->accept(this);
        text_ << (ext->op() == tok::plus ? " += " : " -= ");
        out->accept(this);
        text_.end_line(";");
    }

    text_.decrease_indentation();
    text_.add_line("}"); // end block tail loop

    //text_.add_line("STOP_PROFILE");
    decrease_indentation();

    aliased_output_ = false;
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

void CPrinter::visit(AssignmentExpression *e) {
    e->lhs()->accept(this);
    text_ << " = ";
    e->rhs()->accept(this);
}

void CPrinter::visit(PowBinaryExpression *e) {
    text_ << "std::pow(";
    e->lhs()->accept(this);
    text_ << ", ";
    e->rhs()->accept(this);
    text_ << ")";
}

void CPrinter::visit(BinaryExpression *e) {
    auto pop = parent_op_;
    // TODO unit tests for parenthesis and binops
    bool use_brackets =
        Lexer::binop_precedence(pop) > Lexer::binop_precedence(e->op())
        || (pop==tok::divide && e->op()==tok::times);
    parent_op_ = e->op();

    auto lhs = e->lhs();
    auto rhs = e->rhs();
    if(use_brackets) {
        text_ << "(";
    }
    lhs->accept(this);
    switch(e->op()) {
        case tok::minus :
            text_ << "-";
            break;
        case tok::plus :
            text_ << "+";
            break;
        case tok::times :
            text_ << "*";
            break;
        case tok::divide :
            text_ << "/";
            break;
        case tok::lt     :
            text_ << "<";
            break;
        case tok::lte    :
            text_ << "<=";
            break;
        case tok::gt     :
            text_ << ">";
            break;
        case tok::gte    :
            text_ << ">=";
            break;
        case tok::equality :
            text_ << "==";
            break;
        default :
            throw compiler_exception(
                "CPrinter unsupported binary operator " + yellow(token_string(e->op())),
                e->location());
    }
    rhs->accept(this);
    if(use_brackets) {
        text_ << ")";
    }

    // reset parent precedence
    parent_op_ = pop;
}

