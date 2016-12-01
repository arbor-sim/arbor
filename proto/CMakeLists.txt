add_executable(gaussian-proto gaussian-proto.cpp)

set_target_properties(gaussian-proto
   PROPERTIES
   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/proto"
)

