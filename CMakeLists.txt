cmake_minimum_required(VERSION 3.20)
project(metalrt VERSION 0.3.0 LANGUAGES CXX)
include(GNUInstallDirs)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

option(METALRT_ENABLE_METAL "Build the native Apple Metal backend" ${APPLE})
option(METALRT_BUILD_TESTS "Build runtime correctness and lifetime tests" ON)
option(METALRT_BUILD_EXAMPLES "Build standalone runtime examples" ON)
option(METALRT_TEST_METAL "Register native Metal tests (requires a GPU)" ${APPLE})
option(METALRT_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(METALRT_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "Native Metal requires macOS and the Apple SDK. Use -DMETALRT_ENABLE_METAL=OFF for reference validation.")
    endif()
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version" FORCE)
    elseif(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 13.0)
        message(FATAL_ERROR "This backend requires macOS 13.0 or later")
    endif()
    enable_language(OBJCXX)
endif()

find_package(Threads REQUIRED)
add_library(metalrt STATIC
    src/runtime/resources.cpp
    src/runtime/device.cpp
    src/runtime/completion.cpp
    src/runtime/ordering.cpp
    src/runtime/validation.cpp
    src/runtime/recording.cpp
    src/runtime/stream.cpp
    src/runtime/scopes.cpp
    src/runtime/graph.cpp
    src/runtime/graph_update.cpp
    src/backends/reference/backend.cpp
    src/helpers/control.cpp
    src/helpers/ring.cpp)
add_library(metalrt::metalrt ALIAS metalrt)
target_compile_features(metalrt PUBLIC cxx_std_17)
target_include_directories(metalrt PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(metalrt PUBLIC Threads::Threads)
set_target_properties(metalrt PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(metalrt PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(METALRT_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(metalrt PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(metalrt PUBLIC -fsanitize=address,undefined)
endif()
if(METALRT_ENABLE_METAL)
    set(metalrt_metal_sources
        src/backends/metal/backend.mm
        src/backends/metal/device.mm
        src/backends/metal/compilation.mm
        src/backends/metal/cache.mm
        src/backends/metal/submission.mm
        src/backends/metal/validation.mm
        src/backends/metal/preparation.mm
        src/backends/metal/encoding.mm
        src/backends/metal/indirect.mm
    )
    target_sources(metalrt PRIVATE ${metalrt_metal_sources})
    target_compile_definitions(metalrt PRIVATE METALRT_HAS_METAL=1)
    set_source_files_properties(${metalrt_metal_sources} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    set_target_properties(metalrt PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED YES)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    target_link_libraries(metalrt PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()

# These are ordinary asset copies. No script converts MSL source into C++ or
# generates a kernel implementation. Source files remain human-editable .metal.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")
foreach(shader dependent_kernels runtime_tests file_dispatch graph_features gpu_control ring_pipeline stable_parameters)
    configure_file("kernels/${shader}.metal" "shaders/${shader}.metal" COPYONLY)
endforeach()

if(METALRT_BUILD_TESTS)
    enable_testing()
    add_executable(test_interop tests/test_interop.cpp)
    target_link_libraries(test_interop PRIVATE metalrt::metalrt)
    add_test(NAME reference_interop COMMAND test_interop)
    set_tests_properties(reference_interop PROPERTIES TIMEOUT 30 LABELS "runtime;reference;interop")
    if(METALRT_ENABLE_METAL)
        add_executable(test_metal_interop tests/test_metal_interop.mm)
        target_link_libraries(test_metal_interop PRIVATE metalrt::metalrt)
        target_compile_options(test_metal_interop PRIVATE -fobjc-arc)
        set_target_properties(test_metal_interop PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED YES)
        if(METALRT_TEST_METAL)
            add_test(NAME metal_interop COMMAND test_metal_interop)
            set_tests_properties(metal_interop PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120 LABELS "runtime;metal;interop")
        endif()
    endif()
    foreach(test runtime shader_file graph_features control ring scope_safety timing)
        add_executable(test_${test} "tests/test_${test}.cpp")
        target_link_libraries(test_${test} PRIVATE metalrt::metalrt)
        target_compile_definitions(test_${test} PRIVATE
            METALRT_TEST_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")
        add_test(NAME reference_${test} COMMAND test_${test})
        set_tests_properties(reference_${test} PROPERTIES TIMEOUT 90 LABELS "runtime;reference")
        if(METALRT_TEST_METAL AND METALRT_ENABLE_METAL)
            add_test(NAME metal_${test} COMMAND test_${test} --metal)
            set_tests_properties(metal_${test} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120 LABELS "runtime;metal")
        endif()
    endforeach()
    target_sources(test_runtime PRIVATE
        tests/runtime/execution.cpp
        tests/runtime/recording.cpp
        tests/runtime/validation.cpp
        tests/runtime/completion.cpp
        tests/runtime/optional_buffers.cpp
        tests/runtime/cache.cpp
    )
    if(METALRT_TEST_METAL AND METALRT_ENABLE_METAL)
        # Shader instrumentation changes the binary pipeline and may prevent
        # archive hits. Validate actual reuse separately from instrumented runs.
        add_test(NAME metal_kernel_cache COMMAND test_runtime --metal-cache)
        set_tests_properties(metal_kernel_cache PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120
            LABELS "runtime;metal" ENVIRONMENT "MTL_SHADER_VALIDATION=0" RUN_SERIAL TRUE)
        set_tests_properties(metal_runtime PROPERTIES RUN_SERIAL TRUE)
    endif()
    add_test(NAME runtime_installed_consumer
        COMMAND ${CMAKE_COMMAND}
            "-DRUNTIME_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DCONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/consumer"
            "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/package-consumer-test"
            "-DTEST_CONFIG=$<CONFIG>"
            "-DTEST_GENERATOR=${CMAKE_GENERATOR}"
            "-DTEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DTEST_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
            "-DTEST_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
            "-DTEST_CTEST=${CMAKE_CTEST_COMMAND}"
            "-DTEST_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test-installed-consumer.cmake")
    set_tests_properties(runtime_installed_consumer PROPERTIES TIMEOUT 120 LABELS "runtime;reference;package")
endif()

if(METALRT_BUILD_EXAMPLES)
    foreach(example dependent_kernels replay_updates parallel_branches gpu_control stream_events overhead_benchmark ring_pipeline stable_parameters)
        add_executable(${example} "examples/${example}.cpp")
        target_link_libraries(${example} PRIVATE metalrt::metalrt)
        target_compile_definitions(${example} PRIVATE
            METALRT_EXAMPLE_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")
        if(METALRT_BUILD_TESTS)
            set(example_arguments --reference)
            if(example STREQUAL "overhead_benchmark")
                list(APPEND example_arguments --iterations 8 --warmup 2)
            elseif(example STREQUAL "dependent_kernels")
                list(APPEND example_arguments --policy indirect)
            endif()
            add_test(NAME example_${example}_reference COMMAND ${example} ${example_arguments})
            set_tests_properties(example_${example}_reference PROPERTIES TIMEOUT 60 LABELS "runtime;reference;example")
        endif()
    endforeach()
    if(METALRT_BUILD_TESTS)
        add_test(NAME example_ring_double_per_slot_reference COMMAND ring_pipeline --reference --double --per-slot)
        set_tests_properties(example_ring_double_per_slot_reference PROPERTIES TIMEOUT 60 LABELS "runtime;reference;example")
    endif()
endif()

include(CMakePackageConfigHelpers)
install(TARGETS metalrt EXPORT metalrt-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES README.md DESTINATION ${CMAKE_INSTALL_DATADIR}/metalrt)
install(FILES docs/compatibility.md DESTINATION ${CMAKE_INSTALL_DATADIR}/metalrt/docs)
install(EXPORT metalrt-targets NAMESPACE metalrt:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metalrt)
configure_package_config_file(cmake/metalrt-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/metalrt-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metalrt)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/metalrt-config-version.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/metalrt-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/metalrt-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metalrt)
