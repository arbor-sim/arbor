cdef class py_recipe:
    cdef recipe* c_recipe
    
    def __cinit__(self, int num_cells):
        self.c_recipe = make_recipe(num_cells).release()
        
    def __dealloc__(self):
        del self.c_recipe

cdef class py_model:
    cdef model* c_model

    def __init__(self, py_recipe recipe, domain_decomposition decomp):
        self.c_model = new module(recipe, decomp)

    def __dealloc__(self):
        del self.c_model

    def reset(self):
      self.c_model.reset()

    def run(self, float tfinal, float dt):
      return self.c_model.run(tfinal, dt)

    def num_spikes(self):
      return self.c_model.num_spikes()
    def num_groups(self):
      return self.c_model.num_groups()
    def num_cells(self):
      return self.c_model.num_cells()
