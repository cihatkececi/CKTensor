cmake_minimum_required(VERSION 3.20)

file(GLOB_RECURSE CKTENSOR_TEST_FILES *.cpp)

add_executable(cktensor_tests ${CKTENSOR_TEST_FILES} catch.hpp)

target_link_libraries(cktensor_tests PRIVATE cktensor)
target_include_directories(cktensor_tests PRIVATE ${CKTENSOR_INCLUDE_DIR})

include(CTest)
include(Catch)
catch_discover_tests(cktensor_tests)
