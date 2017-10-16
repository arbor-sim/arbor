#include "recipe.hpp"
#include <Python.h>
#include "cynestmc.h"

namespace nest {
namespace mc {

class cyrecipe: public recipe {
private:
    PyObject* pyRecipe;
    
protected:
    template<class F>
    void python_call(const F& f) {
        if (! f()) throw PyException();
    }
    
public:
    cyrecipe(PyObject* pyRecipe): pyRecipe(pyRecipe) {
        Py_INCREF(pyRecipe);
    }

    ~cyrecipe() {
        Py_DECREF(pyRecipe);
    }
    
    cell_size_type num_cells() const {
        cell_size_type r;
        python_call([&] {py_num_cells(pyRecipe, r);});
        return r;
    }

    util::unique_any get_cell(cell_gid_type t) const {
        util::unique_any r;
        python_call([&] {py_get_cell(pyRecipe, t, r);});
        return r;
    }
    
    cell_kind get_cell_kind(cell_gid_type t) const {
        cell_kind r;
        python_call([&] {py_get_cell_kind(pyRecipe, t, r);});
        return r;
    }

    cell_count_info get_cell_count_info(cell_gid_type t) {
        cell_count_info r;
        python_call([&] {py_get_cell_count_info(pyRecipe, t, r);});
        return r;
    }
    
    std::vector<cell_connection> connections_on(cell_gid_type t) const {
        std::vector<cell_connection> r;
        python_call([&] {py_connections_on(pyRecipe, t, r);});
        return r;
    }
};

int c_call_something(cyrecipe* recipe) {
    try {return recipe->num_cells();}
    catch (PythonException) {}
}
}
}
