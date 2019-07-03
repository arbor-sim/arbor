#pragma once

#include <string>

#include "cprinter.hpp"
#include "module.hpp"
#include "cexpr_emit.hpp"

std::string emit_cuda_cpp_source(const Module& m, const printer_options& opt);
std::string emit_cuda_cu_source(const Module& m, const printer_options& opt);

class CudaPrinter: public CPrinter {
public:
    CudaPrinter(std::ostream& out): CPrinter(out) {}

    void visit(CallExpression*) override;
    void visit(VariableExpression*) override;
};

