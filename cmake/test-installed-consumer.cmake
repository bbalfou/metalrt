# A package boundary test, not a source or shader generator. Install ONLY this
# component, then compile and execute a separate find_package() consumer.
foreach(required RUNTIME_BINARY_DIR CONSUMER_SOURCE_DIR WORK_DIR TEST_CTEST)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "Missing package-test setting: ${required}")
    endif()
endforeach()

function(run_step description)
    execute_process(COMMAND ${ARGN}
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${description} failed (${result})\n${output}\n${error}")
    endif()
endfunction()

file(REMOVE_RECURSE "${WORK_DIR}")
set(prefix "${WORK_DIR}/prefix")
set(consumer_build "${WORK_DIR}/consumer-build")
run_step("Installing standalone runtime"
    "${CMAKE_COMMAND}" --install "${RUNTIME_BINARY_DIR}"
    --prefix "${prefix}" --config "${TEST_CONFIG}")

foreach(forbidden tiny-metal-nn nlohmann pcg32 json)
    if(EXISTS "${prefix}/${TEST_INSTALL_INCLUDEDIR}/${forbidden}")
        message(FATAL_ERROR "The runtime installation leaked a neural dependency: ${forbidden}")
    endif()
endforeach()

run_step("Configuring independent installed consumer"
    "${CMAKE_COMMAND}" -S "${CONSUMER_SOURCE_DIR}" -B "${consumer_build}"
    -G "${TEST_GENERATOR}"
    "-DCMAKE_PREFIX_PATH=${prefix}"
    "-DCMAKE_CXX_COMPILER=${TEST_CXX_COMPILER}"
    # Keep command-line instrumentation flags consistent with the installed
    # static library (for example a UBSan-only build using CMAKE_CXX_FLAGS).
    "-DCMAKE_CXX_FLAGS=${TEST_CXX_FLAGS}"
    "-DCMAKE_EXE_LINKER_FLAGS=${TEST_EXE_LINKER_FLAGS}"
    "-DCMAKE_BUILD_TYPE=${TEST_CONFIG}")
run_step("Building independent installed consumer"
    "${CMAKE_COMMAND}" --build "${consumer_build}" --config "${TEST_CONFIG}" --parallel 2)
run_step("Executing independent installed consumer"
    "${TEST_CTEST}" --test-dir "${consumer_build}" --build-config "${TEST_CONFIG}" --output-on-failure)
message(STATUS "Standalone runtime package and consumer passed without neural dependencies")
