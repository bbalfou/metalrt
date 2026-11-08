cmake_minimum_required(VERSION 3.20)
project(metalrt_installed_consumer LANGUAGES CXX)
find_package(metalrt 0.2 CONFIG REQUIRED)
add_executable(metalrt_consumer main.cpp)
target_link_libraries(metalrt_consumer PRIVATE metalrt::metalrt)
target_compile_definitions(metalrt_consumer PRIVATE
    CONSUMER_SHADER_FILE="${CMAKE_CURRENT_SOURCE_DIR}/consumer.metal")
enable_testing()
add_test(NAME installed_runtime_reference COMMAND metalrt_consumer)
