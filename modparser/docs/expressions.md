## NMODL file structure

From the perspective of a parser, NMODL files can be considered to be a collection of high-level
blocks. For example, take the following extract from an NMODL file:

    NEURON {
        THREADSAFE
        SUFFIX mymod
        USEION k WRITE ik
        RANGE  gbar, ik, ek
        GLOBAL minf, mtau, hinf, htau
    }

    STATE {
        m h
    }

    PROCEDURE trates(v) {
        LOCAL var
        var=q10^((celsius-22)/10)
        minf=1-1/(1+exp(v/km))
        hinf=1/(1+exp((v-vhalfh)/kh))

        mtau = 0.6
        htau = 1500
    }

This NMODL file has three blocks

1. a NEURON block, which has some information about the module
2. a STATE block, which lists the state variables for this mechanism
3. a PROCEDUCE block, which defines the procedure `trates`

Broadly speaking, the blocks in an NMODL file can be divided into one of two categories

1. **Descriptive Blocks**: blocks such as NEURON and STATE, that describe variables and information
about the mechanisms.
3. **Verb Blocks**: blocks such as PROCEDURE, FUNCTION, INITIAL and BREAKPOINT, which describe
actions on data (hence the verb moniker).

Separating the description of the data from the operations on the data makes sense in some ways.

## Parsing Overview

Parsing NMODL files is performed in three stages

1. **First pass**: parse all of the descriptive blocks and parse the verb blocks without attempting
variable lookup.
2. **Identifier tables**: build tables that allow for lookup of all variables defined in the descriptive
blocks, and register signatures of procedures and functions. The creation of the identifier tables
has to wait until all descriptive blocks have been parsed, because information about some variables
is contained in multiple descroptive blocks, which can be defined in any order.
3. **Identifier lookup**: traverse the AST for verb blocks, looking up all variable and call nodes
in the identifier tables generated in step 2. Also build tables of local variables in each
function/procedure.

After the three stages, we will have the full AST for all of the verb blocks, and descriptions of
the variables and procedures/functions in the mechanism.

## Parsing Routines

###parse_call

###parse_line_expression

###parse_expression

###parse_parenthesis_expression

###parse_binop

###parse_local

###parse_prototype

###parse_high_level
called at the start of a new expression. It tests what type of expression is on the line, e.g. if it
is a  local variable definition or a proper expression.
-  **`tok_local`**: the line starts with LOCAL keyword, which is parsed with **`parse_local`**
-  **`tok_identifier`**: the line starts with an identifier. This could be one of several things,
    including  a procedure call or an assignment. The routine **`parse_line_expression`** is called
    to handle all of these cases.

###parse_primary
When parsing an expression the next node in the AST can have different forms, e.g. a number,
     variable, function call or an expression encolsed in parenthesis. The function `parse_primary`
     tests to determine the type of the next node, then calls the appropriate `parse` routine.

-   **`tok_number`**: return **`parse_number`**
-   **`tok_identifier`**:
    - if followed by **`tok_lparen`**, return **`parse_call`**
    - else, return **`parse_identifier`**
-   **`tok_lparen`**: return **`parse_parenthesis_expression`**

###parse_number
Currently only double-precision floating point numbers are supported. When a number is encountered
by the lexer, it is stored as a `std::string` as it was encountered in the input stream. The
**`tok_number`** token is converted into an expression, the string representation is converted to
`double` for internal representation in the **`NumberExpression`**. If support for numeric types other
than `double` was added, the test could be performed on the string, and the appropriate numeric type
chosen here.

###parse_identifier
Parses a valid identifier token. An identifier is used to either name a variable or a
function/procedure. Identifiers follow the familier rules

-   start with either alpha or underscore `[_a-zA-Z]`
    -   can not start with a number
-   remaining characters can include alphanumeric and underscore, i.e. `[_a-zA-Z0-9]`
-   can't clash with predefined keywords, e.g. LOCAL, GLOBAL, etc.

**`parse_identifier`** saves the current identifier token, **`tok_identifier`**, as an
**`IdentifierExpression`** node.






