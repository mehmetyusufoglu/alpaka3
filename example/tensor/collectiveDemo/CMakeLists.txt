cmake_minimum_required(VERSION 3.25)
project(tensorCollectiveDemo LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSupport.cmake)

add_executable(tensorCollectiveDemo src/tensorCollectiveDemo.cpp)
target_link_libraries(tensorCollectiveDemo PUBLIC alpaka::alpaka)
alpaka_finalize(tensorCollectiveDemo)

alpaka_add_tensor_cuda_support(tensorCollectiveDemo)

find_package(MPI QUIET)
if(MPI_FOUND)
    if(TARGET MPI::MPI_CXX)
        set(_alpaka_tensor_collective_mpi_target MPI::MPI_CXX)
    elseif(TARGET MPI::MPI_C)
        set(_alpaka_tensor_collective_mpi_target MPI::MPI_C)
    endif()

    if(DEFINED _alpaka_tensor_collective_mpi_target)
        target_link_libraries(tensorCollectiveDemo PUBLIC ${_alpaka_tensor_collective_mpi_target})
        target_compile_definitions(tensorCollectiveDemo PRIVATE ALPAKA_TENSOR_COLLECTIVE_DEMO_HAS_MPI)
    endif()
endif()

set_target_properties(tensorCollectiveDemo PROPERTIES FOLDER example)
