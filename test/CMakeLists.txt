message(STATUS "Building tests for ${PROJECT_NAME}")

# Only find test files in THIS project's test directory, not dependencies
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.f90")

# Filter to ensure we only get files from this specific directory
list(FILTER TEST_SOURCES INCLUDE REGEX "^${CMAKE_CURRENT_SOURCE_DIR}/test_.*\\.f90$")

# Separate benchmark programs from correctness tests
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*benchmark.*\\.f90$")
file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_benchmark_*.f90")

message(STATUS "Test sources found: ${TEST_SOURCES}")

# Build each correctness test
foreach(test_file ${TEST_SOURCES})
  # Extract the filename without extension
  get_filename_component(test_name ${test_file} NAME_WE)

  message(STATUS "Found test: ${test_name}")

  add_executable(${test_name} ${test_file})

  # Specify the include directories
  target_include_directories(${test_name} PRIVATE "${CMAKE_BUILD_PREFIX}" "${CMAKE_BUILD_PREFIX}/mod")

  target_link_libraries(${test_name} PRIVATE diffstruc)

  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# These cases must reject invalid handles rather than dereference stale nodes.
add_test(NAME test_context_stale COMMAND test_context stale)
set_tests_properties(test_context_stale PROPERTIES WILL_FAIL TRUE)
add_test(NAME test_context_guard COMMAND test_context guard)
set_tests_properties(test_context_guard PROPERTIES WILL_FAIL TRUE)

# Build benchmark programs (not registered as ctest targets)
foreach(bench_file ${BENCH_SOURCES})
  get_filename_component(bench_name ${bench_file} NAME_WE)
  message(STATUS "Found benchmark: ${bench_name}")
  add_executable(${bench_name} ${bench_file})
  target_include_directories(${bench_name} PRIVATE "${CMAKE_BUILD_PREFIX}" "${CMAKE_BUILD_PREFIX}/mod")
  target_link_libraries(${bench_name} PRIVATE diffstruc)
endforeach()
