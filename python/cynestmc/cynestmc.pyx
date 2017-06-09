cdef class Recipe:
    cdef CRecipe* ptr
    
    def __cinit__(self, int left, int right):
        self.ptr = new CRecipe(left, right)

    def __dealloc__(self):
        del self.ptr

cdef class Model:
    cdef CModel* ptr
    cdef domain_decomposition* decomp

    def __cinit__(self, Recipe recipe):
        cdef group_rules r
        r.policy = use_multicore
        r.target_group_size = 1
        
        self.decomp = new domain_decomposition(recipe.ptr[0], r)
        self.ptr = new CModel(recipe.ptr[0], self.decomp[0])

    def __dealloc__(self):
        del self.ptr
        del self.decomp

    def reset(self):
        self.ptr.reset()

    def run(self, float tfinal, float dt):
        return self.ptr.run(tfinal, dt)

    def num_spikes(self):
        return self.ptr.num_spikes()
