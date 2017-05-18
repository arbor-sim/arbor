# Configures and installs all files needed to perform
# a full NestMC regression that validates generated spikes.

if(NMC_WITH_CUDA)
    SET(REG_TEST_N_G "-g 1000")
else()
    SET(REG_TEST_N_G "-g 1")
endif()


# Configure the bash script so it contains the correct absolute paths
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate.sh.in ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate.sh )

# Copy the configured runner to the build dir
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# Copy the python script doing the comparison
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# Copy the blessed output data
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# Add a the "make regression_test" command
add_custom_target(regression_test
${CMAKE_CURRENT_BINARY_DIR}/validate.sh 
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Perform three NestMC runs with known output and compare" VERBATIM
)

