cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 23)


file(GLOB_RECURSE CKTENSOR_EXAMPLE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach (file ${CKTENSOR_EXAMPLE_FILES})
    get_filename_component(exe_name ${file} NAME_WE)
    add_executable(${exe_name} ${file} )
    target_link_libraries(${exe_name} PRIVATE cktensor)
endforeach (file ${CKTENSOR_EXAMPLE_FILES})
