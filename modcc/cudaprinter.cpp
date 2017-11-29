#include <algorithm>
#include <string>
#include <unordered_set>

#include "cexpr_emit.hpp"
#include "cudaprinter.hpp"
#include "lexer.hpp"

std::string CUDAPrinter::pack_name() {
    return module_name_ + "_ParamPack";
}

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

    module_name_ = module_->module_name();

    //
    // Implementation header.
    //
    // Contains the parameter pack and protypes of c wrappers around cuda kernels.
    //

    set_buffer(impl_interface_);

    // headers
    buffer().add_line("#pragma once");
    buffer().add_line("#include <backends/event.hpp>");
    buffer().add_line("#include <backends/fvm_types.hpp>");
    buffer().add_line("#include <backends/multi_event_stream_state.hpp>");
    buffer().add_line("#include <backends/gpu/kernels/detail.hpp>");
    buffer().add_line("#include <util/simple_table.hpp>");
    buffer().add_line();

    buffer().add_line("namespace arb { namespace gpu{");
    buffer().add_line("using deliverable_event_stream_state = multi_event_stream_state<deliverable_event_data>;");
    buffer().add_line();

    // definition of parameter pack type
    std::vector<std::string> param_pack;
    buffer().add_gutter() << "struct " << pack_name()  << " {";
    buffer().end_line();
    buffer().increase_indentation();
    buffer().add_line("using T = arb::fvm_value_type;");
    buffer().add_line("using I = arb::fvm_size_type;");
    buffer().add_line("// array parameters");
    for(auto const &var: array_variables) {
        buffer().add_line("T* " + var->name() + ";");
        param_pack.push_back(var->name() + ".data()");
    }
    buffer().add_line("// scalar parameters");
    for(auto const &var: scalar_variables) {
        buffer().add_line("T " + var->name() + ";");
        param_pack.push_back(var->name());
    }
    buffer().add_line("// ion channel dependencies");
    for(auto& ion: m.neuron_block().ions) {
        auto tname = "ion_" + ion.name;
        for(auto& field : ion.read) {
            buffer().add_line("T* ion_" + field.spelling + ";");
            param_pack.push_back(tname + "." + field.spelling + ".data()");
        }
        for(auto& field : ion.write) {
            buffer().add_line("T* ion_" + field.spelling + ";");
            param_pack.push_back(tname + "." + field.spelling + ".data()");
        }
        buffer().add_line("I* ion_" + ion.name + "_idx_;");
        param_pack.push_back(tname + ".index.data()");
    }

    buffer().add_line("// cv index to cell mapping and cell time states");
    buffer().add_line("const I* ci;");
    buffer().add_line("const T* vec_t;");
    buffer().add_line("const T* vec_t_to;");
    buffer().add_line("const T* vec_dt;");
    param_pack.push_back("vec_ci_.data()");
    param_pack.push_back("vec_t_.data()");
    param_pack.push_back("vec_t_to_.data()");
    param_pack.push_back("vec_dt_.data()");

    buffer().add_line("// voltage and current state within the cell");
    buffer().add_line("T* vec_v;");
    buffer().add_line("T* vec_i;");
    param_pack.push_back("vec_v_.data()");
    param_pack.push_back("vec_i_.data()");

    buffer().add_line("// node index information");
    buffer().add_line("I* ni;");
    buffer().add_line("unsigned long n_;");
    buffer().decrease_indentation();
    buffer().add_line("};");
    buffer().add_line();
    param_pack.push_back("node_index_.data()");
    param_pack.push_back("node_index_.size()");

    // kernel wrapper prototypes
    for(auto const &var: m.symbols()) {
        if (auto e = var.second->is_api_method()) {
            buffer().add_line(APIMethod_prototype(e) + ";");
        }
        else if (var.second->is_net_receive()) {
            buffer().add_line(
                "void deliver_events_" + module_name_ +"(" + pack_name() + " params_, arb::fvm_size_type mech_id, deliverable_event_stream_state state);");
        }
    }
    if(module_->write_backs().size()) {
        buffer().add_line("void write_back_"+module_name_+"("+pack_name()+" params_);");
    }
    buffer().add_line();
    buffer().add_line("}} // namespace arb::gpu");

    //
    // Implementation
    //

    set_buffer(impl_);

    // kernels
    buffer().add_line("#include \"" + module_name_ + "_gpu_impl.hpp\"");
    buffer().add_line();
    buffer().add_line("#include <backends/gpu/intrinsics.hpp>");
    buffer().add_line("#include <backends/gpu/kernels/reduce_by_key.hpp>");
    buffer().add_line();
    buffer().add_line("namespace arb { namespace gpu{");
    buffer().add_line("namespace kernels {");
    buffer().increase_indentation();
    {
        // forward declarations of procedures
        for(auto const &var: m.symbols()) {
            if( var.second->kind()==symbolKind::procedure &&
                var.second->is_procedure()->kind() == procedureKind::normal)
            {
                print_device_function_prototype(var.second->is_procedure());
                buffer().end_line(";");
                buffer().add_line();
            }
        }

        // print stubs that call API method kernels that are defined in the
        // kernels::name namespace
        for(auto const &var: m.symbols()) {
            if (var.second->kind()==symbolKind::procedure &&
                is_in(var.second->is_procedure()->kind(),
                      {procedureKind::normal, procedureKind::api, procedureKind::net_receive}))
            {
                auto e = var.second->is_procedure();
                e->accept(this);
            }
        }
    }

    // print the write_back kernel
    if(module_->write_backs().size()) {
        buffer().add_line("__global__");
        buffer().add_line("void write_back_"+module_name_+"("+pack_name()+" params_) {");
        buffer().increase_indentation();
        buffer().add_line("using value_type = arb::fvm_value_type;");

        buffer().add_line("auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;");
        buffer().add_line("auto const n_ = params_.n_;");
        buffer().add_line("if(tid_<n_) {");
        buffer().increase_indentation();

        for (auto& w: module_->write_backs()) {
            auto& src = w.source_name;
            auto& tgt = w.target_name;

            auto idx = src + "_idx_";
            buffer().add_line("auto "+idx+" = params_.ion_"+to_string(w.ion_kind)+"_idx_[tid_];");
            buffer().add_line("// 1/10 magic number due to unit normalisation");
            buffer().add_line("params_."+tgt+"["+idx+"] = value_type(0.1)*params_.weights_[tid_]*params_."+src+"[tid_];");
        }
        buffer().decrease_indentation(); buffer().add_line("}");
        buffer().decrease_indentation(); buffer().add_line("}");
    }

    buffer().decrease_indentation();
    buffer().add_line("} // kernel namespace");

    // implementation of the kernel wrappers
    buffer().add_line();
    for(auto const &var : m.symbols()) {
        if (auto e = var.second->is_api_method()) {
            buffer().add_line(APIMethod_prototype(e) + " {");
            buffer().increase_indentation();
            buffer().add_line("auto n = params_.n_;");
            buffer().add_line("constexpr int blockwidth = 128;");
            buffer().add_line("dim3 dim_block(blockwidth);");
            buffer().add_line("dim3 dim_grid(impl::block_count(n, blockwidth));");
            buffer().add_line("arb::gpu::kernels::"+e->name()+"_"+module_name_+"<<<dim_grid, dim_block>>>(params_);");
            buffer().decrease_indentation();
            buffer().add_line("}");
            buffer().add_line();
        }
        else if (var.second->is_net_receive()) {
            buffer().add_line("void deliver_events_" + module_name_
                + "(" + pack_name() + " params_, arb::fvm_size_type mech_id, deliverable_event_stream_state state) {");
            buffer().increase_indentation();
            buffer().add_line("const int n = state.n;");
            buffer().add_line("constexpr int blockwidth = 128;");
            buffer().add_line("const auto nblock = impl::block_count(n, blockwidth);");
            buffer().add_line("arb::gpu::kernels::deliver_events<<<nblock, blockwidth>>>(params_, mech_id, state);");
            buffer().decrease_indentation();
            buffer().add_line("}");
            buffer().add_line();
        }
    }

    // add the write_back kernel wrapper if required by this module
    if(module_->write_backs().size()) {
        buffer().add_line("void write_back_"+module_name_+"("+pack_name()+" params_) {");
        buffer().increase_indentation();
        buffer().add_line("auto n = params_.n_;");
        buffer().add_line("constexpr int blockwidth = 128;");
        buffer().add_line("dim3 dim_block(blockwidth);");
        buffer().add_line("dim3 dim_grid(impl::block_count(n, blockwidth));");
        buffer().add_line("arb::gpu::kernels::write_back_"+module_name_+"<<<dim_grid, dim_block>>>(params_);");
        buffer().decrease_indentation();
        buffer().add_line("}");
        buffer().add_line();
    }
    buffer().add_line("}} // namespace arb::gpu");

    //
    // Interface header
    //
    // Included in the front-end C++ code.
    //

    set_buffer(interface_);

    buffer().add_line("#pragma once");
    buffer().add_line();
    buffer().add_line("#include <cmath>");
    buffer().add_line("#include <limits>");
    buffer().add_line();
    buffer().add_line("#include <mechanism.hpp>");
    buffer().add_line("#include <algorithms.hpp>");
    buffer().add_line("#include <backends/event.hpp>");
    buffer().add_line("#include <backends/fvm_types.hpp>");
    buffer().add_line("#include <backends/gpu/multi_event_stream.hpp>");
    buffer().add_line("#include <util/pprintf.hpp>");
    buffer().add_line();
    buffer().add_line("#include \"" + module_name_ + "_gpu_impl.hpp\"");
    buffer().add_line();

    buffer().add_line("namespace arb { namespace gpu{");
    buffer().add_line();

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    std::string class_name = "mechanism_" + module_name_;

    buffer().add_line("template <typename Backend>");
    buffer().add_line("class " + class_name + " : public mechanism<Backend> {");
    buffer().add_line("public:");
    buffer().increase_indentation();
    buffer().add_line("using base = mechanism<Backend>;");
    buffer().add_line("using typename base::value_type;");
    buffer().add_line("using typename base::size_type;");
    buffer().add_line("using typename base::array;");
    buffer().add_line("using typename base::view;");
    buffer().add_line("using typename base::iarray;");
    buffer().add_line("using host_iarray = typename Backend::host_iarray;");
    buffer().add_line("using typename base::iview;");
    buffer().add_line("using typename base::const_iview;");
    buffer().add_line("using typename base::const_view;");
    buffer().add_line("using typename base::ion_type;");
    buffer().add_line("using deliverable_event_stream_state = typename base::deliverable_event_stream_state;");
    buffer().add_line("using param_pack_type = " + pack_name() + ";");

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    for(auto& ion: m.neuron_block().ions) {
        auto tname = "Ion" + ion.name;
        buffer().add_line("struct " + tname + " {");
        buffer().increase_indentation();
        for(auto& field : ion.read) {
            buffer().add_line("view " + field.spelling + ";");
        }
        for(auto& field : ion.write) {
            buffer().add_line("view " + field.spelling + ";");
        }
        buffer().add_line("iarray index;");
        buffer().add_line("std::size_t memory() const { return sizeof(size_type)*index.size(); }");
        buffer().add_line("std::size_t size() const { return index.size(); }");
        buffer().decrease_indentation();
        buffer().add_line("};");
        buffer().add_line(tname + " ion_" + ion.name + ";");
        buffer().add_line();
    }

    //////////////////////////////////////////////
    // constructor
    //////////////////////////////////////////////

    int num_vars = array_variables.size();
    buffer().add_line();
    buffer().add_line(class_name + "(size_type mech_id, const_iview vec_ci, const_view vec_t, const_view vec_t_to, const_view vec_dt, view vec_v, view vec_i, array&& weights, iarray&& node_index):");
    buffer().add_line("   base(mech_id, vec_ci, vec_t, vec_t_to, vec_dt, vec_v, vec_i, std::move(node_index))");
    buffer().add_line("{");
    buffer().increase_indentation();
    buffer().add_gutter() << "size_type num_fields = " << num_vars << ";";
    buffer().end_line();

    buffer().add_line();
    buffer().add_line("// calculate the padding required to maintain proper alignment of sub arrays");
    buffer().add_line("auto alignment  = data_.alignment();");
    buffer().add_line("auto field_size_in_bytes = sizeof(value_type)*size();");
    buffer().add_line("auto remainder  = field_size_in_bytes % alignment;");
    buffer().add_line("auto padding    = remainder ? (alignment - remainder)/sizeof(value_type) : 0;");
    buffer().add_line("auto field_size = size()+padding;");

    buffer().add_line();
    buffer().add_line("// allocate memory");
    buffer().add_line("data_ = array(field_size*num_fields, std::numeric_limits<value_type>::quiet_NaN());");

    // assign the sub-arrays
    // replace this : data_(1*n, 2*n);
    //    with this : data_(1*field_size, 1*field_size+n);

    buffer().add_line();
    buffer().add_line("// asign the sub-arrays");
    for(int i=0; i<num_vars; ++i) {
        char namestr[128];
        sprintf(namestr, "%-15s", array_variables[i]->name().c_str());
        buffer().add_line(
            array_variables[i]->name() + " = data_("
            + std::to_string(i) + "*field_size, " + std::to_string(i+1) + "*field_size);");
    }
    buffer().add_line();

    for(auto const& var : array_variables) {
        double val = var->value();
        // only non-NaN fields need to be initialized, because data_
        // is NaN by default
        if(val == val) {
            buffer().add_line("memory::fill(" + var->name() + ", " + std::to_string(val) + ");");
        }
    }
    buffer().add_line();

    // copy in the weights
    buffer().add_line("// add the user-supplied weights for converting from current density");
    buffer().add_line("// to per-compartment current in nA");
    buffer().add_line("if (weights.size()) {");
    buffer().increase_indentation();
    buffer().add_line("memory::copy(weights, weights_(0, size()));");
    buffer().decrease_indentation();
    buffer().add_line("}");
    buffer().add_line("else {");
    buffer().increase_indentation();
    buffer().add_line("memory::fill(weights_, 1.0);");
    buffer().decrease_indentation();
    buffer().add_line("}");
    buffer().add_line();

    buffer().decrease_indentation();
    buffer().add_line("}");
    buffer().add_line();

    //////////////////////////////////////////////
    //////////////////////////////////////////////

    buffer().add_line("using base::size;");
    buffer().add_line();

    buffer().add_line("std::size_t memory() const override {");
    buffer().increase_indentation();
    buffer().add_line("auto s = std::size_t{0};");
    buffer().add_line("s += data_.size()*sizeof(value_type);");
    for(auto& ion: m.neuron_block().ions) {
        buffer().add_line("s += ion_" + ion.name + ".memory();");
    }
    buffer().add_line("return s;");
    buffer().decrease_indentation();
    buffer().add_line("}");
    buffer().add_line();

    // print the member funtion that packs up the parameters for use on the GPU
    buffer().add_line("void set_params() override {");
    buffer().increase_indentation();
    buffer().add_line("param_pack_ =");
    buffer().increase_indentation();
    buffer().add_line("param_pack_type {");
    buffer().increase_indentation();
    for(auto& str: param_pack) {
        buffer().add_line(str + ",");
    }
    buffer().decrease_indentation();
    buffer().add_line("};");
    buffer().decrease_indentation();
    buffer().decrease_indentation();
    buffer().add_line("}");
    buffer().add_line();

    // name member function
    buffer().add_line("std::string name() const override {");
    buffer().increase_indentation();
    buffer().add_line("return \"" + module_name_ + "\";");
    buffer().decrease_indentation();
    buffer().add_line("}");
    buffer().add_line();

    std::string kind_str = m.kind() == moduleKind::density
                            ? "mechanismKind::density"
                            : "mechanismKind::point";
    buffer().add_line("mechanismKind kind() const override {");
    buffer().increase_indentation();
    buffer().add_line("return " + kind_str + ";");
    buffer().decrease_indentation();
    buffer().add_line("}");
    buffer().add_line();

    // Implement mechanism::set_weights method
    buffer().add_line("void set_weights(array&& weights) override {");
    buffer().increase_indentation();
    buffer().add_line("memory::copy(weights, weights_(0, size()));");
    buffer().decrease_indentation();
    buffer().add_line("}");
    buffer().add_line();

    //////////////////////////////////////////////
    //  print ion channel interface
    //////////////////////////////////////////////

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
    buffer().add_line("typename base::ion_spec uses_ion(ionKind k) const override {");
    buffer().increase_indentation();
    buffer().add_line("bool uses = false;");
    buffer().add_line("bool writes_ext = false;");
    buffer().add_line("bool writes_int = false;");
    for (auto k: {ionKind::Na, ionKind::Ca, ionKind::K}) {
        if (module_->has_ion(k)) {
            auto ion = *module_->find_ion(k);
            buffer().add_line("if (k==ionKind::" + ion.name + ") {");
            buffer().increase_indentation();
            buffer().add_line("uses = true;");
            if (ion.writes_concentration_int()) buffer().add_line("writes_int = true;");
            if (ion.writes_concentration_ext()) buffer().add_line("writes_ext = true;");
            buffer().decrease_indentation();
            buffer().add_line("}");
        }
    }
    buffer().add_line("return {uses, writes_int, writes_ext};");
    buffer().decrease_indentation();
    buffer().add_line("}");
    buffer().add_line();

    // void set_ion(ionKind k, ion_type& i) override
    buffer().add_line("void set_ion(ionKind k, ion_type& i, std::vector<size_type>const& index) override {");
    buffer().increase_indentation();
    for (auto k: {ionKind::Na, ionKind::Ca, ionKind::K}) {
        if (module_->has_ion(k)) {
            auto ion = *module_->find_ion(k);
            buffer().add_line("if (k==ionKind::" + ion.name + ") {");
            buffer().increase_indentation();
            auto n = ion.name;
            auto pre = "ion_"+n;
            buffer().add_line(pre+".index = memory::make_const_view(index);");
            if (ion.uses_current())
                buffer().add_line(pre+".i"+n+" = i.current();");
            if (ion.uses_rev_potential())
                buffer().add_line(pre+".e"+n+" = i.reversal_potential();");
            if (ion.uses_concentration_int())
                buffer().add_line(pre+"."+n+"i = i.internal_concentration();");
            if (ion.uses_concentration_ext())
                buffer().add_line(pre+"."+n+"o = i.external_concentration();");
            buffer().add_line("return;");
            buffer().decrease_indentation();
            buffer().add_line("}");
        }
    }
    buffer().add_line("throw std::domain_error(arb::util::pprintf(\"mechanism % does not support ion type\\n\", name()));");
    buffer().decrease_indentation();
    buffer().add_line("}");
    buffer().add_line();

    //////////////////////////////////////////////
    //////////////////////////////////////////////
    for(auto const &var : m.symbols()) {
        if( var.second->kind()==symbolKind::procedure &&
            var.second->is_procedure()->kind()==procedureKind::api)
        {
            auto proc = var.second->is_api_method();
            auto name = proc->name();
            buffer().add_line("void " + name + "() {");
            buffer().increase_indentation();
            buffer().add_line("arb::gpu::"+name+"_"+module_name_+"(param_pack_);");
            buffer().decrease_indentation();
            buffer().add_line("}");
            buffer().add_line();
        }
        else if( var.second->kind()==symbolKind::procedure &&
                 var.second->is_procedure()->kind()==procedureKind::net_receive)
        {
            // Override `deliver_events`.
            buffer().add_line("void deliver_events(const deliverable_event_stream_state& events) override {");
            buffer().increase_indentation();

            buffer().add_line("arb::gpu::deliver_events_"+module_name_
                              +"(param_pack_, mech_id_, events);");

            buffer().decrease_indentation();
            buffer().add_line("}");
            buffer().add_line();
        }
    }

    if(module_->write_backs().size()) {
        buffer().add_line("void write_back() override {");
        buffer().increase_indentation();
        buffer().add_line("arb::gpu::write_back_"+module_name_+"(param_pack_);");
        buffer().decrease_indentation(); buffer().add_line("}");
    }
    buffer().add_line();

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

    if (!instance_param_ids.empty() || !state_ids.empty()) {
        buffer().add_line("view base::* field_view_ptr(const char* id) const override {");
        buffer().increase_indentation();
        buffer().add_line("static const std::pair<const char*, view "+class_name+"::*> field_tbl[] = {");
        buffer().increase_indentation();
        for (const auto& id: instance_param_ids) {
            auto var = id.token.spelling;
            buffer().add_line("{\""+var+"\", &"+class_name+"::"+var+"},");
        }
        for (const auto& id: state_ids) {
            auto var = id.token.spelling;
            buffer().add_line("{\""+var+"\", &"+class_name+"::"+var+"},");
        }
        buffer().decrease_indentation();
        buffer().add_line("};");
        buffer().add_line();
        buffer().add_line("auto* pptr = util::table_lookup(field_tbl, id);");
        buffer().add_line("return pptr? static_cast<view base::*>(*pptr): nullptr;");
        buffer().decrease_indentation();
        buffer().add_line("}");
        buffer().add_line();
    }

    if (!global_param_ids.empty()) {
        buffer().add_line("value_type base::* field_value_ptr(const char* id) const override {");
        buffer().increase_indentation();
        buffer().add_line("static const std::pair<const char*, value_type "+class_name+"::*> field_tbl[] = {");
        buffer().increase_indentation();
        for (const auto& id: global_param_ids) {
            auto var = id.token.spelling;
            buffer().add_line("{\""+var+"\", &"+class_name+"::"+var+"},");
        }
        buffer().decrease_indentation();
        buffer().add_line("};");
        buffer().add_line();
        buffer().add_line("auto* pptr = util::table_lookup(field_tbl, id);");
        buffer().add_line("return pptr? static_cast<value_type base::*>(*pptr): nullptr;");
        buffer().decrease_indentation();
        buffer().add_line("}");
        buffer().add_line();
    }
    //////////////////////////////////////////////
    //////////////////////////////////////////////

    buffer().add_line("array data_;");
    for(auto var: array_variables) {
        buffer().add_line("view " + var->name() + ";");
    }
    for(auto var: scalar_variables) {
        double val = var->value();
        // test the default value for NaN
        // useful for error propogation from bad initial conditions
        if(val==val) {
            buffer().add_line("value_type " + var->name() + " = " + std::to_string(val) + ";");
        }
        else {
            // the cuda compiler has a bug that doesn't allow initialization of
            // class members with std::numer_limites<>. So simply set to zero.
            buffer().add_line("value_type " + var->name() + " = value_type{0};");
        }
    }

    buffer().add_line("using base::mech_id_;");
    buffer().add_line("using base::vec_ci_;");
    buffer().add_line("using base::vec_t_;");
    buffer().add_line("using base::vec_t_to_;");
    buffer().add_line("using base::vec_dt_;");
    buffer().add_line("using base::vec_v_;");
    buffer().add_line("using base::vec_i_;");
    buffer().add_line("using base::node_index_;");
    buffer().add_line();
    buffer().add_line("param_pack_type param_pack_;");
    buffer().decrease_indentation();
    buffer().add_line("};");
    buffer().add_line();
    buffer().add_line("}} // namespaces");
}

void CUDAPrinter::visit(Expression *e) {
    throw compiler_exception(
        "CUDAPrinter doesn't know how to print " + e->to_string(),
        e->location());
}

void CUDAPrinter::visit(LocalDeclaration *e) {
}

void CUDAPrinter::visit(NumberExpression *e) {
    cexpr_emit(e, buffer().text(), this);
}

void CUDAPrinter::visit(IdentifierExpression *e) {
    e->symbol()->accept(this);
}

void CUDAPrinter::visit(Symbol *e) {
    buffer() << e->name();
}

void CUDAPrinter::visit(VariableExpression *e) {
    buffer() << "params_." << e->name();
    if(e->is_range()) {
        buffer() << "[" << index_string(e) << "]";
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
    else if(s->is_cell_indexed_variable()) {
        return "cid_";
    }
    return "";
}

void CUDAPrinter::visit(IndexedVariable *e) {
    buffer() << "params_." << e->index_name() << "[" << index_string(e) << "]";
}

void CUDAPrinter::visit(CellIndexedVariable *e) {
    buffer() << "params_." << e->index_name() << "[" << index_string(e) << "]";
}


void CUDAPrinter::visit(LocalVariable *e) {
    std::string const& name = e->name();
    buffer() << name;
}

void CUDAPrinter::visit(UnaryExpression *e) {
    cexpr_emit(e, buffer().text(), this);
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
                buffer().add_line("value_type " + var.first + ";");
            }
        }
    }

    // ------------- statements ------------- //
    for(auto& stmt : e->statements()) {
        if(stmt->is_local_declaration()) continue;
        // these all must be handled
        buffer().add_gutter();
        stmt->accept(this);
        if (not stmt->is_if()) {
            buffer().end_line(";");
        }
    }
}

void CUDAPrinter::visit(IfExpression *e) {
    // for now we remove the brackets around the condition because
    // the binary expression printer adds them, and we want to work
    // around the -Wparentheses-equality warning
    buffer() << "if(";
    e->condition()->accept(this);
    buffer() << ") {\n";
    buffer().increase_indentation();
    e->true_branch()->accept(this);
    buffer().decrease_indentation();
    buffer().add_line("}");
    // check if there is a false-branch, i.e. if
    // there is an "else" branch to print
    if (auto fb = e->false_branch()) {
        buffer().add_gutter() << "else ";
        // use recursion for "else if"
        if (fb->is_if()) {
            fb->accept(this);
        }
        // otherwise print the "else" block
        else {
            buffer() << "{\n";
            buffer().increase_indentation();
            fb->accept(this);
            buffer().decrease_indentation();
            buffer().add_line("}");
        }
    }
}

void CUDAPrinter::print_device_function_prototype(ProcedureExpression *e) {
    buffer().add_line("__device__");
    buffer().add_gutter() << "void " << e->name()
                     << "(" << module_name_ << "_ParamPack const& params_,"
                     << "const int tid_";
    for(auto& arg : e->args()) {
        buffer() << ", arb::fvm_value_type " << arg->is_argument()->name();
    }
    buffer() << ")";
}

void CUDAPrinter::visit(ProcedureExpression *e) {
    // error: semantic analysis has not been performed
    if(!e->scope()) { // error: semantic analysis has not been performed
        throw compiler_exception(
            "CUDAPrinter attempt to print Procedure " + e->name()
            + " for which semantic analysis has not been performed",
            e->location());
    }

    if(e->kind() != procedureKind::net_receive) {
        // print prototype
        print_device_function_prototype(e);
        buffer().end_line(" {");

        // print body
        buffer().increase_indentation();

        buffer().add_line("using value_type = arb::fvm_value_type;");
        buffer().add_line();

        e->body()->accept(this);

        // close up
        buffer().decrease_indentation();
        buffer().add_line("}");
        buffer().add_line();
    }
    else {
        // net_receive() kernel is a special case, not covered by APIMethod visit.

        // Core `net_receive` kernel is called device-side from `kernel::deliver_events`.
        buffer().add_line(       "__device__");
        buffer().add_gutter() << "void net_receive(const " << module_name_ << "_ParamPack& params_, "
                           << "arb::fvm_size_type i_, arb::fvm_value_type weight) {";
        buffer().add_line();
        buffer().increase_indentation();

        buffer().add_line("using value_type = arb::fvm_value_type;");
        buffer().add_line();

        buffer().add_line("auto tid_ = i_;");
        buffer().add_line("auto gid_ __attribute__((unused)) = params_.ni[tid_];");
        buffer().add_line("auto cid_ __attribute__((unused)) = params_.ci[gid_];");

        print_APIMethod_body(e);

        buffer().decrease_indentation();
        buffer().add_line("}");
        buffer().add_line();

        // Global one-thread wrapper for `net_receive` kernel is used to implement the
        // `mechanism::net_receive` method. This is not called in the normal course
        // of event delivery.
        buffer().add_line(       "__global__");
        buffer().add_gutter() << "void net_receive_global("
                           << module_name_ << "_ParamPack params_, "
                           << "arb::fvm_size_type i_, arb::fvm_value_type weight) {";
        buffer().add_line();
        buffer().increase_indentation();

        buffer().add_line("if (threadIdx.x || blockIdx.x) return;");
        buffer().add_line("net_receive(params_, i_, weight);");

        buffer().decrease_indentation();
        buffer().add_line("}");
        buffer().add_line();

        buffer().add_line(       "__global__");
        buffer().add_gutter() << "void deliver_events("
                           << module_name_ << "_ParamPack params_, "
                           << "arb::fvm_size_type mech_id, deliverable_event_stream_state state) {";
        buffer().add_line();
        buffer().increase_indentation();

        buffer().add_line("auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;");
        buffer().add_line("auto const ncell_ = state.n;");
        buffer().add_line();
        buffer().add_line("if(tid_<ncell_) {");
        buffer().increase_indentation();


        buffer().add_line("auto begin = state.ev_data+state.begin_offset[tid_];");
        buffer().add_line("auto end = state.ev_data+state.end_offset[tid_];");
        buffer().add_line("for (auto p = begin; p<end; ++p) {");
        buffer().increase_indentation();
        buffer().add_line("if (p->mech_id==mech_id) {");
        buffer().increase_indentation();
        buffer().add_line("net_receive(params_, p->mech_index, p->weight);");
        buffer().decrease_indentation();
        buffer().add_line("}");
        buffer().decrease_indentation();
        buffer().add_line("}");

        buffer().decrease_indentation();
        buffer().add_line("}");

        buffer().decrease_indentation();
        buffer().add_line("}");
        buffer().add_line();
    }
}

std::string CUDAPrinter::APIMethod_prototype(APIMethod *e) {
    return "void " + e->name() + "_" + module_name_
        + "(" + pack_name() + " params_)";
}

void CUDAPrinter::visit(APIMethod *e) {
    // print prototype
    buffer().add_line("__global__");
    buffer().add_line(APIMethod_prototype(e) + " {");

    if(!e->scope()) { // error: semantic analysis has not been performed
        throw compiler_exception(
            "CUDAPrinter attempt to print APIMethod " + e->name()
            + " for which semantic analysis has not been performed",
            e->location());
    }
    buffer().increase_indentation();

    buffer().add_line("using value_type = arb::fvm_value_type;");
    buffer().add_line();

    buffer().add_line("auto tid_ = threadIdx.x + blockDim.x*blockIdx.x;");
    buffer().add_line("auto const n_ = params_.n_;");
    buffer().add_line();
    buffer().add_line("if(tid_<n_) {");
    buffer().increase_indentation();

    buffer().add_line("auto gid_ __attribute__((unused)) = params_.ni[tid_];");
    buffer().add_line("auto cid_ __attribute__((unused)) = params_.ci[gid_];");

    print_APIMethod_body(e);

    buffer().decrease_indentation();
    buffer().add_line("}");

    buffer().decrease_indentation();
    buffer().add_line("}");
    buffer().add_line();
}

void CUDAPrinter::print_APIMethod_body(ProcedureExpression* e) {
    // load indexes of ion channels
    auto uses_k  = false;
    auto uses_na = false;
    auto uses_ca = false;
    for(auto &symbol : e->scope()->locals()) {
        auto ch = symbol.second->is_local_variable()->ion_channel();
        if(!uses_k   && (uses_k  = (ch == ionKind::K)) ) {
            buffer().add_line("auto kid_  = params_.ion_k_idx_[tid_];");
        }
        if(!uses_ca  && (uses_ca = (ch == ionKind::Ca)) ) {
            buffer().add_line("auto caid_ = params_.ion_ca_idx_[tid_];");
        }
        if(!uses_na  && (uses_na = (ch == ionKind::Na)) ) {
            buffer().add_line("auto naid_ = params_.ion_na_idx_[tid_];");
        }
    }

    // shadows for indexed arrays
    for(auto &symbol : e->scope()->locals()) {
        auto var = symbol.second->is_local_variable();
        if(is_input(var)) {
            auto ext = var->external_variable();
            buffer().add_gutter() << "value_type ";
            var->accept(this);
            buffer() << " = ";
            ext->accept(this);
            buffer().end_line("; // indexed load");
        }
        else if (is_output(var)) {
            buffer().add_gutter() << "value_type " << var->name() << ";";
            buffer().end_line();
        }
    }

    buffer().add_line();
    buffer().add_line("// the kernel computation");

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
            buffer().add_line();
            buffer().add_line("// stores to indexed global memory");
            has_outputs = true;
        }
        buffer().add_gutter();
        if(!is_point_process()) {
            out->accept(this);
            buffer() << (out->op()==tok::plus ? " += " : " -= ");
            in->accept(this);
        }
        else {
            buffer() << "arb::gpu::reduce_by_key(";
            if (out->op()==tok::minus) buffer() << "-";
            in->accept(this);
            // reduce_by_key() takes a pointer to the start of the target
            // array as a parameter. This requires writing the index_name of out, which
            // we can safely assume is an indexed_variable by this point.
            buffer() << ", params_." << out->is_indexed_variable()->index_name() << ", gid_)";
        }
        buffer().end_line(";");
    }

    return;
}

void CUDAPrinter::visit(CallExpression *e) {
    buffer() << e->name() << "(params_, tid_";
    for(auto& arg: e->args()) {
        buffer() << ", ";
        arg->accept(this);
    }
    buffer() << ")";
}

void CUDAPrinter::visit(BinaryExpression *e) {
    cexpr_emit(e, buffer().text(), this);
}
