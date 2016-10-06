#include <algorithm>

#include "cudaprinter.hpp"
#include "lexer.hpp"

/******************************************************************************
******************************************************************************/

CUDAPrinter::CUDAPrinter(Module &m, bool o)
    :   module_(&m)
{
    // make a list of vector types, both parameters and assigned
    // and a list of all scalar types
    std::vector<VariableExpression*> scalar_variables;
    std::vector<VariableExpression*> array_variables;
    for(auto& sym: m.symbols()) {
        if(sym.second->kind()==symbolKind::variable) {
            auto var = sym.second->is_variable();
            if(var->is_range()) {
                array_variables.push_back(var);
            }
            else {
                scalar_variables.push_back(var) ;
            }
        }
    }

    //////////////////////////////////////////////
    // header files
    //////////////////////////////////////////////
    text_.add_line("#pragma once");
    text_.add_line();
    text_.add_line("#include <cmath>");
    text_.add_line("#include <limits>");
    text_.add_line();
    text_.add_line("#include <mechanism.hpp>");
    text_.add_line("#include <mechanism_interface.hpp>");
    //text_.add_line("#include <gpu/util.hpp>");
    text_.add_line();


    text_.add_line("namespace nest{ namespace mc{ namespace mechanisms{ namespace gpu{ namespace " + m.name() + "{");
    text_.add_line();
    increase_indentation();

    ////////////////////////////////////////////////////////////
    // generate the parameter pack
    ////////////////////////////////////////////////////////////
    std::vector<std::string> param_pack;
    text_.add_line("template <typename T, typename I>");
    text_.add_gutter() << "struct " << m.name() << "_ParamPack {";
    text_.end_line();
    text_.increase_indentation();
    text_.add_line("// array parameters");
    for(auto const &var: array_variables) {
        text_.add_line("T* " + var->name() + ";");
        param_pack.push_back(var->name() + ".data()");
    }
    text_.add_line("// scalar parameters");
    for(auto const &var: scalar_variables) {
        text_.add_line("T " + var->name() + ";");
        param_pack.push_back(var->name());
    }
    text_.add_line("// ion channel dependencies");
    for(auto& ion: m.neuron_block().ions) {
        auto tname = "ion_" + ion.name;
        for(auto& field : ion.read) {
            text_.add_line("T* ion_" + field.spelling + ";");
            param_pack.push_back(tname + "." + field.spelling + ".data()");
        }
        for(auto& field : ion.write) {
            text_.add_line("T* ion_" + field.spelling + ";");
            param_pack.push_back(tname + "." + field.spelling + ".data()");
        }
        text_.add_line("I* ion_" + ion.name + "_idx_;");
        param_pack.push_back(tname + ".index.data()");
    }

    text_.add_line("// voltage and current state within the cell");
    text_.add_line("T* vec_v;");
    text_.add_line("T* vec_i;");
    param_pack.push_back("vec_v_.data()");
    param_pack.push_back("vec_i_.data()");

    text_.add_line("T* vec_area;");
    param_pack.push_back("vec_area_.data()");

    text_.add_line("// node index information");
    text_.add_line("I* ni;");
    text_.add_line("unsigned long n_;");
    text_.decrease_indentation();
    text_.add_line("};");
    text_.add_line();
    param_pack.push_back("node_index_.data()");
    param_pack.push_back("node_index_.size()");

    ////////////////////////////////////////////////////////
    // write the CUDA kernels
    ////////////////////////////////////////////////////////
    text_.add_line("namespace kernels {");
    {
        increase_indentation();

        text_.add_line("__device__");
        text_.add_line("inline double atomicAdd(double* address, double val) {");
        text_.increase_indentation();
        text_.add_line("using I = unsigned long long int;");
        text_.add_line("I* address_as_ull = (I*)address;");
        text_.add_line("I old = *address_as_ull, assumed;");
        text_.add_line("do {");
        text_.add_line("assumed = old;");
        text_.add_line("old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val+__longlong_as_double(assumed)));");
        text_.add_line("} while (assumed != old);");
        text_.add_line("return __longlong_as_double(old);");
        text_.decrease_indentation();
        text_.add_line("}");
        text_.add_line();
        /*
        text_.add_line("__device__");
        text_.add_line("inline double atomicSub(double* address, double val) {");
        text_.increase_indentation();
        text_.add_line("return atomicAdd(address, -val);");
        text_.decrease_indentation();
        text_.add_line("}");
        text_.add_line();
        text_.add_line("__device__");
        text_.add_line("inline float atomicSub(float* address, float val) {");
        text_.increase_indentation();
        text_.add_line("return atomicAdd(address, -val);");
        text_.decrease_indentation();
        text_.add_line("}");
        text_.add_line();
        */

        // forward declarations of procedures
        for(auto const &var : m.symbols()) {
            if(   var.second->kind()==symbolKind::procedure
            && var.second->is_procedure()->kind() == procedureKind::normal)
            {
                print_procedure_prototype(var.second->is_procedure());
                text_.end_line(";");
                text_.add_line();
            }
        }

        // print stubs that call API method kernels that are defined in the
        // kernels::name namespace
        auto proctest = [] (procedureKind k) {return k == procedureKind::normal
                                                  || k == procedureKind::api;   };
        for(auto const &var : m.symbols()) {
            if (var.second->kind()==symbolKind::procedure &&
                proctest(var.second->is_procedure()->kind()))
            {
                var.second->accept(this);
            }
        }
        decrease_indentation();
    }
    text_.add_line("} // namespace kernels");
    text_.add_line();

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    std::string class_name = "mechanism_" + m.name();

    text_.add_line("template<typename T, typename I>");
    text_.add_line("class " + class_name + " : public ::nest::mc::mechanisms::gpu::mechanism<T, I> {");
    text_.add_line("public:");
    text_.increase_indentation();
    text_.add_line("using base = ::nest::mc::mechanisms::gpu::mechanism<T, I>;");
    text_.add_line("using value_type  = typename base::value_type;");
    text_.add_line("using size_type   = typename base::size_type;");
    text_.add_line("using vector_type = typename base::vector_type;");
    text_.add_line("using view_type   = typename base::view_type;");
    text_.add_line("using index_type  = typename base::index_type;");
    text_.add_line("using index_view  = typename base::index_view;");
    text_.add_line("using const_index_view  = typename base::const_index_view;");
    text_.add_line("using indexed_view_type= typename base::indexed_view_type;");
    text_.add_line("using ion_type = typename base::ion_type;");
    text_.add_line("using param_pack_type = " + m.name() + "_ParamPack<T,I>;");

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
        text_.add_line();
    }

    //////////////////////////////////////////////
    // constructor
    //////////////////////////////////////////////

    int num_vars = array_variables.size();
    text_.add_line();
    text_.add_line("template <typename IVT>");
    text_.add_line(class_name + "(view_type vec_v, view_type vec_i, IVT node_index) :");
    text_.add_line("   base(vec_v, vec_i, node_index)");
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
        text_.add_line(
            array_variables[i]->name() + " = data_("
            + std::to_string(i) + "*field_size, " + std::to_string(i+1) + "*field_size);");
    }

    for(auto const& var : array_variables) {
        double val = var->value();
        // only non-NaN fields need to be initialized, because data_
        // is NaN by default
        if(val == val) {
            text_.add_line(var->name() + "(memory::all) = " + std::to_string(val) + ";");
        }
    }

    text_.add_line();
    text_.decrease_indentation();
    text_.add_line("}");

    //////////////////////////////////////////////
    //////////////////////////////////////////////

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

    // print the member funtion that
    //   *  sets time step parameters
    //   *  packs up the parameters for use on the GPU
    text_.add_line("void set_params(value_type t_, value_type dt_) override {");
    text_.increase_indentation();
    text_.add_line("t = t_;");
    text_.add_line("dt = dt_;");
    text_.add_line("param_pack_ =");
    text_.increase_indentation();
    text_.add_line("param_pack_type {");
    text_.increase_indentation();
    for(auto& str: param_pack) {
        text_.add_line(str + ",");
    }
    text_.decrease_indentation();
    text_.add_line("};");
    text_.decrease_indentation();
    text_.decrease_indentation();
    text_.add_line("}");
    text_.add_line();

    // name member function
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

    //////////////////////////////////////////////
    //  print ion channel interface
    //////////////////////////////////////////////
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

    auto proctest = [] (procedureKind k) {return k == procedureKind::api;};
    for(auto const &var : m.symbols()) {
        if(   var.second->kind()==symbolKind::procedure
        && proctest(var.second->is_procedure()->kind()))
        {
            auto proc = var.second->is_api_method();
            auto name = proc->name();
            text_.add_line("void " + name + "() {");
            text_.increase_indentation();
            text_.add_line("auto n = size();");
            text_.add_line("auto thread_dim = 192;");
            text_.add_line("dim3 dim_block(thread_dim);");
            text_.add_line("dim3 dim_grid(n/dim_block.x + (n%dim_block.x ? 1 : 0) );");
            text_.add_line();
            text_.add_line(
                "kernels::" + name + "<T,I>"
                + "<<<dim_grid, dim_block>>>(param_pack_);");
            text_.decrease_indentation();
            text_.add_line("}");
            text_.add_line();
        }
    }

    //////////////////////////////////////////////
    //////////////////////////////////////////////

    text_.add_line("vector_type data_;");
    for(auto var: array_variables) {
        text_.add_line("view_type " + var->name() + ";");
    }
    for(auto var: scalar_variables) {
        double val = var->value();
        // test the default value for NaN
        // useful for error propogation from bad initial conditions
        if(val==val) {
            text_.add_line("value_type " + var->name() + " = " + std::to_string(val) + ";");
        }
        else {
            // the cuda compiler has a bug that doesn't allow initialization of
            // class members with std::numer_limites<>. So simply set to zero.
            text_.add_line("value_type " + var->name() + " = value_type{0};");
        }
    }

    text_.add_line("using base::vec_v_;");
    text_.add_line("using base::vec_i_;");
    text_.add_line("using base::vec_area_;");
    text_.add_line("using base::node_index_;");
    text_.add_line();
    text_.add_line("param_pack_type param_pack_;");
    decrease_indentation();
    text_.add_line("};");
    decrease_indentation();
    text_.add_line("}}}}} // namespaces");
}

void CUDAPrinter::visit(Expression *e) {
    throw compiler_exception(
        "CUDAPrinter doesn't know how to print " + e->to_string(),
        e->location());
}

void CUDAPrinter::visit(LocalDeclaration *e) {
}

void CUDAPrinter::visit(NumberExpression *e) {
    text_ << " " << e->value();
}

void CUDAPrinter::visit(IdentifierExpression *e) {
    e->symbol()->accept(this);
}

void CUDAPrinter::visit(Symbol *e) {
    text_ << e->name();
}

void CUDAPrinter::visit(VariableExpression *e) {
    text_ << "params_." << e->name();
    if(e->is_range()) {
        text_ << "[" << index_string(e) << "]";
    }
}

std::string CUDAPrinter::index_string(Symbol *s) {
    if(s->is_variable()) {
        return "tid_";
    }
    else if(auto var = s->is_indexed_variable()) {
        switch(var->ion_channel()) {
            case ionKind::none :
                return "gid_";
            case ionKind::Ca   :
                return "caid_";
            case ionKind::Na   :
                return "naid_";
            case ionKind::K    :
                return "kid_";
            // a nonspecific ion current should never be indexed: it is
            // local to a mechanism
            case ionKind::nonspecific:
                break;
            default :
                throw compiler_exception(
                    "CUDAPrinter unknown ion type",
                    s->location());
        }
    }
    return "";
}

void CUDAPrinter::visit(IndexedVariable *e) {
    text_ << "params_." << e->index_name() << "[" << index_string(e) << "]";
}

void CUDAPrinter::visit(LocalVariable *e) {
    std::string const& name = e->name();
    text_ << name;
}

void CUDAPrinter::visit(UnaryExpression *e) {
    auto b = (e->expression()->is_binary()!=nullptr);
    switch(e->op()) {
        case tok::minus :
            // place a space in front of minus sign to avoid invalid
            // expressions of the form : (v[i]--67)
            // use parenthesis if expression is a binop, otherwise
            // -(v+2) becomes -v+2
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
                "CUDAPrinter unsupported unary operator " + yellow(token_string(e->op())),
                e->location());
    }
}

void CUDAPrinter::visit(BlockExpression *e) {
    // ------------- declare local variables ------------- //
    // only if this is the outer block
    if(!e->is_nested()) {
        for(auto& var : e->scope()->locals()) {
            auto sym = var.second.get();
            // input variables are declared earlier, before the
            // block body is printed
            if(is_stack_local(sym) && !is_input(sym)) {
                text_.add_line("value_type " + var.first + ";");
            }
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

void CUDAPrinter::visit(IfExpression *e) {
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

void CUDAPrinter::print_procedure_prototype(ProcedureExpression *e) {
    text_.add_gutter() << "template <typename T, typename I>\n";
    text_.add_line("__device__");
    text_.add_gutter() << "void " << e->name()
                       << "(" << module_->name() << "_ParamPack<T,I> const& params_,"
                       << "const int tid_";
    for(auto& arg : e->args()) {
        text_ << ", T " << arg->is_argument()->name();
    }
    text_ << ")";
}

void CUDAPrinter::visit(ProcedureExpression *e) {
    // error: semantic analysis has not been performed
    if(!e->scope()) { // error: semantic analysis has not been performed
        throw compiler_exception(
            "CUDAPrinter attempt to print Procedure " + e->name()
            + " for which semantic analysis has not been performed",
            e->location());
    }

    // ------------- print prototype ------------- //
    print_procedure_prototype(e);
    text_.end_line(" {");

    // ------------- print body ------------- //
    increase_indentation();

    text_.add_line("using value_type = T;");
    text_.add_line("using index_type = I;");
    text_.add_line();

    e->body()->accept(this);

    // ------------- close up ------------- //
    decrease_indentation();
    text_.add_line("}");
    text_.add_line();
    return;
}

void CUDAPrinter::visit(APIMethod *e) {
    // ------------- print prototype ------------- //
    text_.add_gutter() << "template <typename T, typename I>\n";
    text_.add_line(       "__global__");
    text_.add_gutter() << "void " << e->name()
                       << "(" << module_->name() << "_ParamPack<T,I> params_) {";
    text_.add_line();

    if(!e->scope()) { // error: semantic analysis has not been performed
        throw compiler_exception(
            "CUDAPrinter attempt to print APIMethod " + e->name()
            + " for which semantic analysis has not been performed",
            e->location());
    }
    increase_indentation();

    text_.add_line("using value_type = T;");
    text_.add_line("using index_type = I;");
    text_.add_line();

    text_.add_line("auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;");
    text_.add_line("auto const n_ = params_.n_;");
    text_.add_line();
    text_.add_line("if(tid_<n_) {");
    increase_indentation();

    text_.add_line("auto gid_ __attribute__((unused)) = params_.ni[tid_];");

    print_APIMethod_body(e);

    decrease_indentation();
    text_.add_line("}");

    decrease_indentation();
    text_.add_line("}\n");
}

void CUDAPrinter::print_APIMethod_body(APIMethod* e) {
    // load indexes of ion channels
    auto uses_k  = false;
    auto uses_na = false;
    auto uses_ca = false;
    for(auto &symbol : e->scope()->locals()) {
        auto ch = symbol.second->is_local_variable()->ion_channel();
        if(!uses_k   && (uses_k  = (ch == ionKind::K)) ) {
            text_.add_line("auto kid_  = params_.ion_k_idx_[tid_];");
        }
        if(!uses_ca  && (uses_ca = (ch == ionKind::Ca)) ) {
            text_.add_line("auto caid_ = params_.ion_ca_idx_[tid_];");
        }
        if(!uses_na  && (uses_na = (ch == ionKind::Na)) ) {
            text_.add_line("auto naid_ = params_.ion_na_idx_[tid_];");
        }
    }

    // shadows for indexed arrays
    for(auto &symbol : e->scope()->locals()) {
        auto var = symbol.second->is_local_variable();
        if(is_input(var)) {
            auto ext = var->external_variable();
            text_.add_gutter() << "value_type ";
            var->accept(this);
            text_ << " = ";
            ext->accept(this);
            text_.end_line("; // indexed load");
        }
        else if (is_output(var)) {
            text_.add_gutter() << "value_type " << var->name() << ";";
            text_.end_line();
        }
    }

    text_.add_line();
    text_.add_line("// the kernel computation");

    e->body()->accept(this);

    // insert stores here
    // take care to use atomic operations for the updates for point processes, where
    // more than one thread may try add/subtract to the same memory location
    auto has_outputs = false;
    for(auto &symbol : e->scope()->locals()) {
        auto in  = symbol.second->is_local_variable();
        auto out = in->external_variable();
        if(out==nullptr || !is_output(in)) continue;
        if(!has_outputs) {
            text_.add_line();
            text_.add_line("// stores to indexed global memory");
            has_outputs = true;
        }
        text_.add_gutter();
        if(!is_point_process()) {
            out->accept(this);
            text_ << (out->op()==tok::plus ? " += " : " -= ");
            in->accept(this);
        }
        else {
            text_ << (out->op()==tok::plus ? "atomicAdd" : "atomicSub") << "(&";
            out->accept(this);
            text_ << ", ";
            in->accept(this);
            text_ << ")";
        }
        text_.end_line(";");
    }

    return;
}

void CUDAPrinter::visit(CallExpression *e) {
    text_ << e->name() << "<T,I>(params_, tid_";
    for(auto& arg: e->args()) {
        text_ << ", ";
        arg->accept(this);
    }
    text_ << ")";
}

void CUDAPrinter::visit(AssignmentExpression *e) {
    e->lhs()->accept(this);
    text_ << " = ";
    e->rhs()->accept(this);
}

void CUDAPrinter::visit(PowBinaryExpression *e) {
    text_ << "std::pow(";
    e->lhs()->accept(this);
    text_ << ", ";
    e->rhs()->accept(this);
    text_ << ")";
}

void CUDAPrinter::visit(BinaryExpression *e) {
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
                "CUDAPrinter unsupported binary operator " + yellow(token_string(e->op())),
                e->location());
    }
    rhs->accept(this);
    if(use_brackets) {
        text_ << ")";
    }

    // reset parent precedence
    parent_op_ = pop;
}

