#
# Copyright 2024 Alpaka Group
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(_TARGET_NAME gemmRP)
project(${_TARGET_NAME} LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)

    if(alpaka_USE_SOURCE_TREE)
        # Don't build the examples recursively
        set(alpaka_BUILD_EXAMPLES OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

add_executable(${_TARGET_NAME} 
    src/gemmRP.cpp
)
target_include_directories(${_TARGET_NAME} PRIVATE src)
target_link_libraries(${_TARGET_NAME} PUBLIC alpaka::alpaka)
alpaka_finalize(${_TARGET_NAME})

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER example)

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} --m 128 --n 128 --k 128 --type float --repeats 2)
