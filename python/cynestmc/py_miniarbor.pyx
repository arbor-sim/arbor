from libc.stdlib cimport malloc, free

cdef extern from "miniapp.hpp":
    cdef int main(int , char**) except +


def mainfunction(argv):
    cdef int argc = <int> len(argv)
    cdef char** argv_chars = <char**> malloc((argc+1) * sizeof(char*))
    main(argc, argv_chars)
    free(argv_chars)



