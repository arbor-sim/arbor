.. _contribcodingstyle:

Coding Guidelines
=================

Python
------

We follow the `black <https://black.readthedocs.io/en/stable/index.html>`__
coding style. It is enforced by an automated check on each pull request. You can
run the following commands to apply it:

.. code::

   # Install the formatter if not present
   pip install black
   # Automatically apply style to a certain file. If unsure what this does read on.
   black . scripts/arbor/build-catalogue.in

The formatter can also be run with ``--check`` to list offending files and
``--diff`` to preview changes. Most editors can `integrate with black
<https://black.readthedocs.io/en/stable/integrations/editors.html>`__.

C++
---

The main development language of Arbor is C++. For Arbor we start with
the community guidelines set out in the `C++ Core
Guidelines <http://isocpp.github.io/CppCoreGuidelines/>`__. These
guidelines are quite generic, and only give loose guidelines for some
important topics like variable naming and indentation.

This wiki will describe the specific extensions and differences to the
C++ Core Guidelines - variable naming. - code formatting (indentation,
placement of curly braces, ``int const&`` vs ``const int&`` etc.) -
rules for topics not covered in the Core Guidelines (e.g. CUDA). -
exceptions to the rules given in the Core Guidelines. - rules for the
CMake build system and directory structure. - rules for external
dependencies (e.g. Boost).

.. TODO::
    This page needs revision.

Code organisation
-----------------

Source files naming conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. TODO:: describe the public/private header and source code organization.

-  ``.cpp`` extension for source files
-  ``.hpp`` extension for header files
-  Names must be lowercase with words separated with underscores.

Namespaces
----------

All code is in ``namespace arb``.

As an example, the ``std::make_unique<>`` function template was not
provided as part of C++11 (it wasn’t introduced until C++14), and we
would like to use it in Arbor. The code sample below shows how
namespaces are declared and formatted in Arbor:

::

   namespace arb {
   namespace util {

   // just because we aren't using C++14, doesn't mean we shouldn't have make_unique
   template <typename T, typename... Args>
   std::unique_ptr<T> make_unique(Args&&... args) {
       return std::unique_ptr<T>(new T(std::forward<Args>(args) ...));
   }

   } // namespace util
   } // namespace arb

In the example above the namespaces are not indented. However,
namespaces should be indented if they are declared in the middle of
code, to make their existance obvious to the person reading the code.

Use an ``impl`` namespace to hide implementation details that should not
be exposed to user space.

You can use ``using`` statements to import individual types or functions
from a namespace - only if it really improves the readability of your
code - only in a function or class scope: don’t pollute namespaces

Formatting statements
---------------------

A lot of the rules here are purely a matter of personal taste, imposed
by the guy who got to set the rules. That said, it follows accepted
practice in the C++ community (if we accept that not everybody has the
same accepted practice), and if followed consistently will make code
easier to understand.

.. code:: c++

   template <typename T>
   class array {
   public:
       using value_type = T;

       value_type& operator[] (std::size_t i) {
           return data_[i];
       }

       const value_type& operator[] (std::size_t i) const {
           return data_[i];
       }

       std::size_t size() const {
           return size_;
       }

   private:
       value_type* data_;
       std::size_t size_;
   };

   // use new lines and indentation to make complex template expressions
   // human readable
   template <
       typename T,
       typename = typename  // assert that T is a built-in arithmetic type
           std::enable_if<
               std::is_arithmetic<T>::value
           >
   >
   T sum(const array<T>& in) {
       return std::accumulate(in.begin(), in.end(), 0);
   }

TODO: When declaring an operator, should we leave a space between the
operator and the following opening parenthesis or should we follow the
convention we use for functions, where we don’t leave a space?

Indentation and whitespace cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  No tabs, 4 spaces
-  Take the extra effort to remove trailing whitespace (at the end of
   the lines and the file).
-  Respect 80-column limit, but go for longer lines when they make sense
   (and make the code clearer)

Variable naming conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All lowercase, words separated by ``_``, but template parameters follow
camel case:

.. code:: c++

   template <typename ValueType>
   class my_class {
   public:
       // ...
   private:
       ValueType val_;
   };

Single letter template parameters should be preferred.

TODO: Or should we force single letter parameters aliased by more
meaningful type names inside the class (either public or private
depending on our intent)?

*Avoid* obfuscated names of old C heritage.

Recurring variables naming conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Some variable names are recurring a lot inside every code. It
would be nice if we could decide on the most common ones.

-  ``count`` or ``cnt``
-  ``index`` or ``idx``
-  ``iter`` or ``it``
-  …

Ben says “depends… I would use ``count`` or ``index`` unless the scope
of the variable is very small. Using ``it`` is standard C++ short hand,
but again for fairly limited scope.”

Member variables
~~~~~~~~~~~~~~~~

Private member variables must be suffixed by ``_``, while public member
variables must not.

TODO: Any conventions about ``static`` variables, ``const``\ s or global
``const``\ s?

Member initialisation lists
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constructors member initialisation lists should be as follows:

.. code:: c++

   // everything goes on one line if clear
   class my_class {
   public:
       my_class(int a):
           a_(a)
       {}

       my_class(int a, int b, int c):
           a_(a), b_(b) , c_(c)
       {}

   private:
       int a_ = 0;
       int b_ = 0;
       int c_ = 0;
   };

   // use one entry per line if multiple lines needed
   class my_class {
   public:
       my_class(int a, int o, int p):
           apple_(a),
           orange_(o),
           pear_(find_pair_type(p))
       {}

   private:
       int apple_;
       int orange_;
       int pear_;
   };

Member functions
~~~~~~~~~~~~~~~~

Make sure to declare ``const`` if it is not changing the object’s state.

Getters and Setters
~~~~~~~~~~~~~~~~~~~

Before filling up a class with getters and setters, consider seriously
if those members are meant actually to be public. If nonetheless getters
and/or setters are needed, don’t use the ``get_`` and ``set_`` prefixes.

.. code:: c++

   template <typename T>
   class my_class {
   public:
       // ...
       T value() const {
           return value_;
       }

       void value(const T& val) {
           // perhaps do something before assigning, otherwise it could be just public
           value_ = val;
       }
   private:
       T value_;
   };

Declaring references and pointers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   // ok
   std::string& s = ...;
   const std::string& s = ...;
   std::string* s = ...;
   const std::string* s = ...;
   std::string* const s = ...;

   // not ok
   std::string &s = ...;
   const std::string &s = ...;
   std::string *s = ...;
   const std::string *s = ...;
   std::string *const s = ...;

Generally, we follow C++’s convention for references and pointers, as it
is the style used in the C++ standard, and also the recommendation of
the `C++ Core Guidelines
NL.18 <http://isocpp.github.io/CppCoreGuidelines/#nl18-use-c-style-declarator-layout>`__.
Precedence and the C++ language grammar may offer some support the other
convention, but not enough support!

Macros
~~~~~~

Macros are C-ish, so they must be avoided. If not possible, they must be
written in capitals, with words separated by underscores.

Always use ``{}``, even for single statement ``if``, ``for``, etc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It makes code clearer, and avoids nasty bugs that occur when
refactoring. It also avoids some errors when merging with git.

::


   // ok
   for (auto& v: vector) {
       // increment the value!
       v++;
   }

   // bad
   for (auto& v: vector)
       // increment the value!
       v++;

don’t put ``{`` on a new line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Except when indentation of arguments or when doing member initialization
in constructors.

::

   // it makes sense to have the { on a new line here for clarity
   std::vector<std::string> foo(
       std::vector<std::vector<int>>& values,
       std::map<int, std::string>& name_table)
   {
       // do some work
   }

leave a space between ``if``, ``for`` etc and following parenthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accords with `K&R
style <http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines.html#Rl-knr>`__,
and makes a visual distinction with function evaluation

::

   // ok
   for (auto& v: vector) {
       v++;
   }

   // not ok
   for(auto& v: vector) {
       v++;
   }

use ``using`` instead of ``typedef``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is easier to read, consistent with ``auto``:

::

   // good
   using int_container = std::vector<int>;

   // bad
   typedef std::vector<int> int_container;

and can be used for template aliases

::

   template <typename T>
   using aligned_container = std::vector<T, my_fancy_aligned_allocator<T>>;

Use scoped enum instead of enum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   // good
   enum class ionKind {sodium, calcium};
   // bad
   enum ionKind {ion_sodium, ion_calcium};

And stick to the naming scheme for all enums of ``xxxKind`` to make it
clear throughout the code whenever an enum is being used, for example:

::

   auto i = current(voltage, ionKind::calcium);

Use ``struct`` for POD wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

But ``class`` if it has any sort of fancy logic associated with it

Memory management
-----------------

use ``unique_ptr``
~~~~~~~~~~~~~~~~~~

Actually, feel free to use naked pointers in your code, but make sure
that you use smart pointers to handle allocation and freeing of memory.
When a developer sees a naked pointer in Arbor they can think “good, I
don’t have to worry about responsibility for freeing that memory”.
Furthermore, if ``unique_ptr`` handles allocation and freeing of memory,
the user doesn’t have to concern themselves with freeing memory ever.

This practice implies that care must be taken to ensure that the
resource managed by a ``unique_ptr`` has to outlive any raw pointers
that are obtained from its ``get()`` member.

while avoiding ``shared_ptr`` whenever possible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you think long and hard, you will probably realise that you actually
want a ``unique_ptr``. Shared pointers have performance overheads, and
are quite easy to misuse. For example by creating circular references
that ironically lead to memory never being freed.

Header files
------------

use pragma once
~~~~~~~~~~~~~~~

Use ``#pragma once`` to guard against including the same header twice.
This might not be completely standard compliant, but it is supported by
every compiler under the sun, and is much cleaner than ``#ifdef``
guards.

don’t rely on headers being included elsewhere
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, if you use ``std::vector<int>`` in a file, make sure to
have ``#include <vector>`` at the top of the source file.

Relying on headers being include elsewhere can lead to portability
problems, for example on OS X you have to ``#include <cmath>`` for some
math functions that are imported via other header files with gcc on
Linux.

Sort headers alphabetically
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To make it easy to search for a header in a long list of includes.

For example:

.. code:: c++

   #include <algorithm>
   #include <fstream>
   #include <map>
   #include <queue>
   #include <set>

use C++ wrappers for C standard headers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   // good
   #include <cmath>
   #include <cstdio>

   // bad
   #include <math.h>
   #include <stdio.h>

when calling C stdlib functions, use the ``std::``-prefix versions,
e.g., ``std::printf(...)`` instead of ``printf``. Most of the times C++
wrappers just bring into ``std`` the C declarations, but sometimes the
wrappers have more syntactic sugar and call the same internal builtins
that their C counterparts call (for example GCC).

group headers together
~~~~~~~~~~~~~~~~~~~~~~

In the following order

1. C++ standard libary
2. system C headers (POSIX, kernel interfaces etc.)
3. external libraries
4. public arbor headers
5. private arbor headers

For example:

.. code:: c++

   // first C++ standard headers
   #include <algorithm>
   #include <fstream>
   #include <map>

   // then system C headers
   #include <signal.h>
   #include <sys/select.h>

   // externals
   #include <vector/Vector.hpp>

   // public arbor headers
   #include <arbor/common_types.hpp>
   #include <arbor/simulation.hpp>

   // private arbor headers (note we use quotes for private project headers).
   #include "cell_group.hpp"
   #include "util/optional.hpp"
