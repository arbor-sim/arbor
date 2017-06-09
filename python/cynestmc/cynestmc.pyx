cdef class Recipe:
    cdef CRecipe* ptr
    
    def __cinit__(self, int left, int right):
        self.ptr = new CRecipe(left, right)

    def __dealloc__(self):
        del self.ptr

cdef class Model:
    cdef CModel* ptr

    def __cinit__(self, Recipe recipe, ...):
        self.ptr = new CModel(recipe.ptr[0], ...)

    def __dealloc__(self):
        del self.ptr

    def reset(self):
        self.ptr.reset()

    def run(self, float tfinal, float dt):
        return self.ptr.run(tfinal, dt)

    def num_spikes(self):
        return self.ptr.num_spikes()
