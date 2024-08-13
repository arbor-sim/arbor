#pragma once

//#ifndef ARB_EXPORT_DEBUG
//#   define ARB_EXPORT_DEBUG
//#endif

#include <arbor/util/visibility.hpp>

/* library build type (ARB_SUP_STATIC_LIBRARY/ARB_SUP_SHARED_LIBRARY) */
#define ARB_SUP_STATIC_LIBRARY

#ifndef ARB_SUP_EXPORTS
#   if defined(arbor_sup_EXPORTS)
        /* we are building arbor-sup dynamically */
#       ifdef ARB_EXPORT_DEBUG
#           pragma message "we are building arbor-sup dynamically"
#       endif
#       define ARB_SUP_API ARB_SYMBOL_EXPORT
#   elif defined(arbor_sup_EXPORTS_STATIC)
        /* we are building arbor-sup statically */
#       ifdef ARB_EXPORT_DEBUG
#           pragma message "we are building arbor-sup statically"
#       endif
#       define ARB_SUP_API
#   else
        /* we are using the library arbor-sup */
#       if defined(ARB_SUP_SHARED_LIBRARY)
            /* we are importing arbor-sup dynamically */
#           ifdef ARB_EXPORT_DEBUG
#              pragma message "we are importing arbor-sup dynamically"
#           endif
#           define ARB_SUP_API ARB_SYMBOL_IMPORT
#       else
            /* we are importing arbor-sup statically */
#           ifdef ARB_EXPORT_DEBUG
#               pragma message "we are importing arbor-sup statically"
#           endif
#           define ARB_SUP_API
#       endif
#   endif
#endif
