cmake_minimum_required(VERSION 3.25)
project(tensorCollectiveFFT LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSupport.cmake)

# NCCL discovery honors CMake cache variables such as ALPAKA_NCCL_LIBRARY/ALPAKA_NCCL_INCLUDE_DIR
# and the NCCL_ROOT environment variable; ensure LD_LIBRARY_PATH reaches libnccl.so at runtime.
# MPI support only activates when CMake finds MPI::MPI_CXX (set MPI_C_COMPILER=mpicc and
# MPI_CXX_COMPILER=mpicxx, or load the MPI module before configuring).
# Example configure: cmake -S .. -B build -DCMAKE_BUILD_TYPE=Release \
#     -DMPI_C_COMPILER=mpicc -DMPI_CXX_COMPILER=mpicxx

add_executable(
    tensorCollectiveFFT
    src/tensorCollectiveFFT.cpp
    src/bootstrap.cpp
    src/dataSource.cpp
    src/options.cpp
    src/verification.cpp
)
target_include_directories(tensorCollectiveFFT PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tensorCollectiveFFT PUBLIC alpaka::alpaka)
alpaka_finalize(tensorCollectiveFFT)

alpaka_add_tensor_cuda_support(tensorCollectiveFFT)

find_package(MPI QUIET COMPONENTS C CXX)
if(MPI_FOUND)
    if(TARGET MPI::MPI_CXX)
        set(_alpaka_tensor_collective_mpi_target MPI::MPI_CXX)
    elseif(TARGET MPI::MPI_C)
        set(_alpaka_tensor_collective_mpi_target MPI::MPI_C)
    endif()

    if(DEFINED _alpaka_tensor_collective_mpi_target)
        target_link_libraries(tensorCollectiveFFT PUBLIC ${_alpaka_tensor_collective_mpi_target})
        target_compile_definitions(tensorCollectiveFFT PRIVATE ALPAKA_TENSOR_COLLECTIVE_DEMO_HAS_MPI)
    else()
        message(
            STATUS
            "MPI discovered but no usable imported target (MPI::MPI_CXX or MPI::MPI_C); "
            "tensorCollectiveFFT will run single-rank only."
        )
    endif()
else()
    message(STATUS "MPI not found; tensorCollectiveFFT will build without multi-rank support.")
endif()

set_target_properties(tensorCollectiveFFT PROPERTIES FOLDER example)
