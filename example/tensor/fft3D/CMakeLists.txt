project(tensorFft3D LANGUAGES CXX)

add_executable(tensorFft3D src/tensorFft3D.cpp)

target_link_libraries(tensorFft3D PRIVATE alpaka::alpaka)

alpaka_finalize(tensorFft3D)
alpaka_add_tensor_cuda_support(tensorFft3D)

set_target_properties(tensorFft3D PROPERTIES FOLDER example)
