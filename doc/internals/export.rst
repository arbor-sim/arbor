.. _export:

Exporting Symbols
=================

The Arbor libraries are compiled with `hidden visibility <https://gcc.gnu.org/wiki/Visibility>`_ by default. Thus, symbols which are part of
the public API need to be marked explicitly as visible. Arbor provides a couple of macros to
annotate functions and classes which are defined in the header file ``export.hpp`` in each library's
include directory, i.e. ``include/arbor/export.hpp``. These header files are generated at configure
time based on the build variant, compiler and platform.

Macro Descripiton
-----------------

.. c:macro:: ARB_LIBNAME_API

    Here ``LIBNAME`` is a placeholder for the library's name: ``ARB_ARBOR_API`` for the main arbor
    library, ``ARB_ARBORIO_API`` for arborio, etc. This macro is intended to annotate functions,
    classes and structs which need to be accessible when using the library. Note, that it expands to
    different values when arbor is being built vs. when arbor is being used by an application. Below
    we list the places where the macro needs to be added or can be safely omitted (we assume all of
    the symbols below are part of the public API).

    .. code-block:: cpp
        :caption: header.hpp

        // free function declaration
        ARB_ARBOR_API void foo();

        // free function (inline)
        void bar(int i) { /* ... */ }

        // function template (inline)
        template<typename T>
        void baz(T i) { /* ... */ }

        // class declaration
        // note: this will make all member symbols visible
        class ARB_ARBOR_API A {
            A();
            friend std::ostream& operator<<(std::ostream& o, A const & a);
        };

        // class (inline)
        class B {
            /* ... */
        };

        // template class (inline)
        template<typename T>
        class C {
            /* ... */
        };

        // (extern) global variable declarations
        ARB_ARBOR_API int g;
        ARB_ARBOR_API extern int h;


    .. code-block:: cpp
        :caption: source.cpp

        // free function definition
        ARB_ARBOR_API void foo() { /* ... */ }

        // class member functions
        A::A() { /* ... */ }

        // friend functions
        ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, A const& a) { /* ... */ }

        // (extern) global variable definitions
        ARB_ARBOR_API int g = 10;
        ARB_ARBOR_API int h = 11;


.. c:macro:: ARB_SYMBOL_VISIBLE

    Objects which are type-erased and passed across the library boundaries sometimes need runtime
    type information (rtti). In particular, exception classes and classes stored in ``std::any`` or
    similar need to have the correct runtime information attached. Hidden visibility strips away
    this information which leads to all kind of unexpected behaviour. Therefore, all such classes
    must be annotated with this macro which guarantees that the symbol is always visible. Note, it
    is not enough to use the first macro for these cases.

    .. code-block:: cpp
        :caption: header.hpp

        // exception class
        class ARB_SYMBOL_VISIBLE some_error : public std::runtime_error {
            /* ... */
        };

        // class D will be type-erased and restored by an any_cast or similar
        class ARB_SYMBOL_VISIBLE D {
            /* ... */
        };

