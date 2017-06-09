
# cdef class Recipe:
#     def num_cells(): pass
#     def get_cell(cgt): pass
#     def get_cell_count_info(cgt): pass
#     def get_cell_count_info(cgt): pass
#     def connections_on(cgt): pass        

def call_something(recipe):
    wrapper = new cyrecipe(recipe)
    return c_call_something(cyrecipe)
    
cdef public cbool py_num_cells(
    PyObject* pyrecipe,
    cell_size_type& r) \
    with gil \
    except False:
    r = (<object>pyrecipe).num_cells()
    return True
    
cdef public cbool py_get_cell(
    PyObject* pyrecipe,
    cell_gid_type t,
    util::unique_any&) \
    with gil \
    except False:
    r = (<object>pyrecipe).get_cell(t)
    return True
                                    
cdef public cbool py_get_cell_kind(
    PyObject* pyrecipe,
    cell_gid_type t,
    cell_kind& r) \
    with gil \
    except False:
    r = (<object>pyrecipe).get_cell(t)
    return True

cdef public cbool py_get_cell_count_info(
    PyObject* pyrecipe,
    cell_gid_type t,
    cell_kind& r) \
    with gil \
    except False:
    r = (<object>pyrecipe).get_cell_count_info(t)
    return True
