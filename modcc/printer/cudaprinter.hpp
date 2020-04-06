#pragma once

#include <string>

#include "cprinter.hpp"
#include "module.hpp"
#include "cexpr_emit.hpp"

std::string emit_gpu_cpp_source(const Module& m, const printer_options& opt);
std::string emit_gpu_cu_source(const Module& m, const printer_options& opt);

class GpuPrinter: public CPrinter {
public:
    GpuPrinter(std::ostream& out): CPrinter(out) {}

    void visit(CallExpression*) override;
    void visit(VariableExpression*) override;
};

