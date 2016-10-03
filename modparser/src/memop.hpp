#pragma once

#include "util.hpp"
#include "lexer.hpp"

/// Defines a memory operation that is to performed by an APIMethod.
/// Kernels can read/write global state via an index, e.g.
///     - loading voltage v from VEC_V in matrix before computation
///     - loading a variable associated with an ionic variable
///     - accumulating an update to VEC_RHS/VEC_D after computation
///     - adding contribution to an ionic current
/// How these operations are handled will vary significantly from
/// one backend implementation to another, so inserting expressions
/// directly into the APIMethod body to perform them is not appropriate.
/// Instead, each API method stores two lists
///     - a list of load/input transactions to perform before kernel
///     - a list of store/output transactions to perform after kernel
/// The lists are of MemOps, which describe the local and external variables
template <typename Symbol>
struct MemOp {
    using symbol_type = Symbol;
    tok op;
    Symbol *local;
    Symbol *external;

    MemOp(tok o, Symbol *loc, Symbol *ext)
    : op(o), local(loc), external(ext)
    {
        const tok valid_ops[] = {tok::plus, tok::minus, tok::eq};
        if(!is_in(op, valid_ops)) {
            throw compiler_exception(
                "invalid operation  for creating a MemOp : " +
                loc->to_string() + yellow(token_string(op)) + ext->to_string(),
                loc->location());
        }
    }
};

