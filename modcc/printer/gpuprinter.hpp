#pragma once

#include <string>

#include "cprinter.hpp"
#include "module.hpp"
#include "cexpr_emit.hpp"
#include <libmodcc/export.hpp>

ARB_LIBMODCC_API std::string emit_gpu_cpp_source(const Module& m, const printer_options& opt);
ARB_LIBMODCC_API std::string emit_gpu_cu_source(const Module& m, const printer_options& opt);

class ARB_LIBMODCC_API GpuPrinter: public CPrinter {
public:
    GpuPrinter(std::ostream& out): CPrinter(out) {}

    void visit(CallExpression*) override;
    void visit(VariableExpression*) override;
    void visit(WhiteNoise*) override;
};

