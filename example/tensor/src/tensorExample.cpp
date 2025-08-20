/* Copyright 2025 Alpaka Tensor Library Contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <iostream>
#include <random>
#include <chrono>

using namespace alpaka;

template<typename T_Cfg>
auto example(T_Cfg const& cfg) -> int
{
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];

    std::cout << "Alpaka Tensor Library Example" << std::endl;
    std::cout << "Backend: " << alpaka::core::demangledName(exec) << std::endl;
    std::cout << "Device: " << deviceSpec.getApi().getName() << std::endl;
    std::cout << std::endl;

    // Select a device and create queue
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = devSelector.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    try {
        // Create 1D tensors for testing
        constexpr std::size_t tensorSize = 1024;
        tensor::Tensor1D<float> tensorA({tensorSize}, tensor::DataType::Float32, tensor::Layout::RowMajor, "tensorA");
        tensor::Tensor1D<float> tensorB({tensorSize}, tensor::DataType::Float32, tensor::Layout::RowMajor, "tensorB");

        std::cout << "Created tensors of size: " << tensorSize << std::endl;
        std::cout << "Tensor A name: " << tensorA.name() << std::endl;
        std::cout << "Tensor B name: " << tensorB.name() << std::endl;

        // Fill tensors with test data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 10.0f);

        for (std::size_t i = 0; i < tensorSize; ++i) {
            tensorA(i) = dist(gen);
            tensorB(i) = dist(gen);
        }

        std::cout << "Filled tensors with random data" << std::endl;
        std::cout << "Sample data - A[0]: " << tensorA(0) << ", B[0]: " << tensorB(0) << std::endl;

        // Test basic tensor operations
        std::cout << "\n--- Testing Tensor Properties ---" << std::endl;
        std::cout << "Tensor A shape: [" << tensorA.shape()[0] << "]" << std::endl;
        std::cout << "Tensor A size: " << tensorA.size() << " elements" << std::endl;
        std::cout << "Tensor A size: " << tensorA.sizeBytes() << " bytes" << std::endl;
        std::cout << "Device allocated: " << (tensorA.isDeviceAllocated() ? "Yes" : "No") << std::endl;

        // Test memory management
        std::cout << "\n--- Testing Memory Management ---" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        tensorA.allocateDevice();
        tensorB.allocateDevice();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Device allocation time: " << duration << " ms" << std::endl;
        std::cout << "Device allocated: " << (tensorA.isDeviceAllocated() ? "Yes" : "No") << std::endl;

        // Test compatibility checks
        std::cout << "\n--- Testing Compatibility ---" << std::endl;
        std::cout << "Shape compatible: " << (tensorA.isShapeCompatible(tensorB) ? "Yes" : "No") << std::endl;
        std::cout << "Device compatible: " << (tensorA.isDeviceCompatible(tensorB) ? "Yes" : "No") << std::endl;

        // Test tensor operations (CPU-only for now)
        std::cout << "\n--- Testing Operations ---" << std::endl;
        
        // Create a result tensor for addition using simplified API
        auto result = tensor::ops::add(tensorA, tensorB);
        
        std::cout << "Addition result - result[0]: " << result(0) << " (expected: " << (tensorA(0) + tensorB(0)) << ")" << std::endl;

        // Test ReLU
        tensor::Tensor1D<float> reluTest({5}, tensor::DataType::Float32, tensor::Layout::RowMajor, "reluTest");
        reluTest(0) = -2.0f;
        reluTest(1) = -1.0f;
        reluTest(2) = 0.0f;
        reluTest(3) = 1.0f;
        reluTest(4) = 2.0f;

        std::cout << "\nBefore ReLU: [" << reluTest(0) << ", " << reluTest(1) << ", " << reluTest(2) 
                  << ", " << reluTest(3) << ", " << reluTest(4) << "]" << std::endl;

        tensor::ops::relu_inplace(reluTest);

        std::cout << "After ReLU:  [" << reluTest(0) << ", " << reluTest(1) << ", " << reluTest(2) 
                  << ", " << reluTest(3) << ", " << reluTest(4) << "]" << std::endl;

        // Test tensor utilities
        std::cout << "\n--- Testing Utilities ---" << std::endl;
        tensor::Tensor1D<float> zeroTensor({10});
        zeroTensor.zero();
        std::cout << "Zero tensor[0]: " << zeroTensor(0) << " (expected: 0)" << std::endl;

        tensor::Tensor1D<float> fillTensor({10});
        fillTensor.fill(3.14f);
        std::cout << "Fill tensor[0]: " << fillTensor(0) << " (expected: 3.14)" << std::endl;

        // Test copy constructor
        std::cout << "\n--- Testing Copy Operations ---" << std::endl;
        tensor::Tensor1D<float> copyTensor = tensorA;
        std::cout << "Copy tensor name: " << copyTensor.name() << std::endl;
        std::cout << "Copy tensor[0]: " << copyTensor(0) << " (original: " << tensorA(0) << ")" << std::endl;

        std::cout << "\n--- Tensor Library Test Completed Successfully ---" << std::endl;
        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

auto main(int argc, char* argv[]) -> int
{
    // Execute the example for each enabled backend
    return executeForEachIfHasDevice(
        [=](auto const& tag) { return example(tag); },
        onHost::allBackends(onHost::enabledApis));
}
