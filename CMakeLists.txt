cmake_minimum_required(VERSION 3.20)
project(cktensor)

set(CMAKE_CXX_STANDARD 23)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    option(USE_ASAN "Use the address sanitizer" OFF)

    if (MSVC)
        add_compile_options(/W4)

        if (USE_ASAN)
            add_compile_options(/fsanitize=address)
            add_link_options(/fsanitize=address)
        endif ()
    else ()
        add_compile_options(-Wall -Wpedantic -Wextra)

        if (USE_ASAN)
            add_compile_options(-fsanitize=address)
            add_link_options(-fsanitize=address)
        endif ()
    endif ()
endif ()

set(MKL_THREADING intel_thread)
find_package(MKL CONFIG)
message(STATUS "${MKL_IMPORTED_TARGETS}")

if (MKL_FOUND)
    add_compile_definitions(CKTENSOR_USE_MKL)
endif (MKL_FOUND)
message(STATUS "MKL_INCLUDE: ${MKL_INCLUDE}")
message(STATUS "MKL_LIBRARIES: ${MKL_LIBRARIES}")

file(GLOB_RECURSE CKTENSOR_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

set(CKTENSOR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)


add_library(cktensor INTERFACE ${CKTENSOR_HEADER_FILES})
target_include_directories(cktensor INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )

if (MKL_FOUND)
    target_compile_options(cktensor INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(cktensor INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(cktensor INTERFACE $<LINK_ONLY:MKL::MKL>)
endif (MKL_FOUND)

# Examples
add_subdirectory(example)

# Install
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS cktensor
        EXPORT cktensorTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include
        )

#export(TARGETS
#        cktensor
#        NAMESPACE cktensor::
#        FILE "${CMAKE_CURRENT_BINARY_DIR}/cktensorConfig.cmake"
#        )

install(EXPORT cktensorTargets
        FILE cktensorTargets.cmake
        NAMESPACE cktensor::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cktensor"
        )

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/cktensorConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cktensor
        )

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/cktensorConfig.cmake"
#        "${CMAKE_CURRENT_BINARY_DIR}/cktensorConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cktensor
        )


# Testing
include(CTest)

if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif ()
