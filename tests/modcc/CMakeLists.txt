set(HEADERS
	${PROJECT_SOURCE_DIR}/modcc/src/constantfolder.hpp
	${PROJECT_SOURCE_DIR}/modcc/src/cprinter.hpp
	${PROJECT_SOURCE_DIR}/modcc/src/expressionclassifier.hpp
	${PROJECT_SOURCE_DIR}/modcc/src/lexer.hpp
	${PROJECT_SOURCE_DIR}/modcc/src/module.hpp
	${PROJECT_SOURCE_DIR}/modcc/src/perfvisitor.hpp
	${PROJECT_SOURCE_DIR}/modcc/src/parser.hpp
	${PROJECT_SOURCE_DIR}/src/util.hpp
	${PROJECT_SOURCE_DIR}/modcc/src/modccutil.hpp
#	${PROJECT_SOURCE_DIR}/modcc/src/variablerenamer.hpp
	test.hpp
)

set(MODCC_TEST_SOURCES
	# unit tests
	test_lexer.cpp
	test_module.cpp
	test_optimization.cpp
	test_parser.cpp
	#test_printers.cpp
	test_visitors.cpp

	# unit test driver
	driver.cpp
)

add_definitions("-DDATADIR=\"${CMAKE_SOURCE_DIR}/data\"")
add_executable(test_modcc ${MODCC_TEST_SOURCES} ${HEADERS})

target_link_libraries(test_modcc LINK_PUBLIC compiler gtest)

set_target_properties(test_modcc
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)
