add_executable(symge-demo symge-demo.cpp)

set_target_properties(symge-demo
   PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/proto"
)

