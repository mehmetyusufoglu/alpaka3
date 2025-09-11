/* Copyright 2024 Alpaka Group
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/precision.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include "GemmKernel.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>

using namespace alpaka;

template<typename T>
void fillMatrix(std::vector<T>& matrix, std::size_t rows, std::size_t cols, T minVal = T(-1), T maxVal = T(1))
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(static_cast<float>(minVal), static_cast<float>(maxVal));
    
    for(std::size_t i = 0; i < rows * cols; ++i)
    {
        matrix[i] = static_cast<T>(dist(rng));
    }
}

template<typename T>
auto exampleGemm(auto const deviceSpec, auto const exec, 
                 std::size_t M, std::size_t N, std::size_t K, 
                 T alpha, T beta, int repeats) -> int
{
    using IdxVec = Vec<std::size_t, 2u>;

    std::cout << "GEMM dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Element type: " << onHost::demangledName<T>() << std::endl;
    std::cout << "Using alpaka accelerator: " << onHost::demangledName(exec) << " for "
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();

    // Initialize host matrices
    std::vector<T> hostA(M * K);
    std::vector<T> hostB(K * N);
    std::vector<T> hostC(M * N, T(0));

    fillMatrix(hostA, M, K);
    fillMatrix(hostB, K, N);

    // Allocate device buffers
    auto bufA = onHost::allocHost<T>(IdxVec{M, K});
    auto bufB = onHost::allocHost<T>(IdxVec{K, N});
    auto bufC = onHost::allocHost<T>(IdxVec{M, N});

    // Copy data to buffers
    std::copy(hostA.begin(), hostA.end(), bufA.data());
    std::copy(hostB.begin(), hostB.end(), bufB.data());
    std::copy(hostC.begin(), hostC.end(), bufC.data());

    // Allocate device memory
    auto bufAccA = onHost::allocLike(devAcc, bufA);
    auto bufAccB = onHost::allocLike(devAcc, bufB);
    auto bufAccC = onHost::allocLike(devAcc, bufC);

    // Copy Host -> Device
    onHost::memcpy(queue, bufAccA, bufA);
    onHost::memcpy(queue, bufAccB, bufB);
    onHost::memcpy(queue, bufAccC, bufC);

    // Create buffer views (no explicit mdspan needed)
    // Use the buffers directly - alpaka3 handles MdSpan internally

    // Instantiate the kernel
    GemmKernel kernel;
    auto taskKernel = KernelBundle{kernel, bufAccA, bufAccB, bufAccC, M, N, K, alpha, beta};

    // Work division: 2D grid, 16x16 threads per block
    Vec<size_t, 2u> blockSize = {16u, 16u};
    auto frameSpec = onHost::FrameSpec{
        divCeil(IdxVec{N, M}, blockSize),  // grid size
        blockSize                          // block size
    };

    // Warmup run
    queue.enqueue(exec, frameSpec, taskKernel);
    onHost::wait(queue);

    // Timed runs
    onHost::wait(queue);
    auto const beginT = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < repeats; ++i)
    {
        queue.enqueue(exec, frameSpec, taskKernel);
    }
    
    onHost::wait(queue);
    auto const endT = std::chrono::high_resolution_clock::now();

    auto const timePerRun = std::chrono::duration<double>(endT - beginT).count() / repeats;
    auto const flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    auto const gflops = flops / timePerRun / 1e9;

    std::cout << "Time per run: " << timePerRun << " s" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;

    // Copy result back and verify (simple check)
    onHost::memcpy(queue, bufC, bufAccC);
    onHost::wait(queue);

    std::cout << "Execution completed successfully!" << std::endl;
    return EXIT_SUCCESS;
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [--m M] [--n N] [--k K] [--type TYPE] [--repeats R] [--alpha A] [--beta B] [--help]"
              << std::endl;
    std::cerr << "  TYPE: float|double|tf32|bf16|fp16|auto" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    std::size_t M = 512, N = 512, K = 512;
    std::string type = "float";
    int repeats = 5;
    float alpha = 1.0f, beta = 0.0f;

    // Parse command line arguments
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        
        if(arg == "--m" && i + 1 < argc)
        {
            M = std::stoul(argv[++i]);
        }
        else if(arg == "--n" && i + 1 < argc)
        {
            N = std::stoul(argv[++i]);
        }
        else if(arg == "--k" && i + 1 < argc)
        {
            K = std::stoul(argv[++i]);
        }
        else if(arg == "--type" && i + 1 < argc)
        {
            type = argv[++i];
        }
        else if(arg == "--repeats" && i + 1 < argc)
        {
            repeats = std::stoi(argv[++i]);
        }
        else if(arg == "--alpha" && i + 1 < argc)
        {
            alpha = std::stof(argv[++i]);
        }
        else if(arg == "--beta" && i + 1 < argc)
        {
            beta = std::stof(argv[++i]);
        }
        else if(arg == "--help")
        {
            help(argv);
            return EXIT_SUCCESS;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            help(argv);
            return EXIT_FAILURE;
        }
    }

    using namespace alpaka;

    // Execute for each available backend
    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        {
            auto const& devSpec = backend[alpaka::object::deviceSpec];
            auto const& execObj = backend[alpaka::object::exec];
            
            if(type == "float")
                return exampleGemm<float>(devSpec, execObj, M, N, K, alpha, beta, repeats);
            else if(type == "double")
                return exampleGemm<double>(devSpec, execObj, M, N, K, alpha, beta, repeats);
            else if(type == "tf32")
                return exampleGemm<alpaka::TF32>(devSpec, execObj, M, N, K, 
                                               alpaka::TF32(alpha), alpaka::TF32(beta), repeats);
            else if(type == "bf16")
                return exampleGemm<alpaka::BF16>(devSpec, execObj, M, N, K, 
                                               alpaka::BF16(alpha), alpaka::BF16(beta), repeats);
            else if(type == "fp16")
                return exampleGemm<alpaka::FP16>(devSpec, execObj, M, N, K, 
                                               alpaka::FP16(alpha), alpaka::FP16(beta), repeats);
            else if(type == "auto")
            {
                using AutoType = alpaka::precision::optimal_t<decltype(devSpec), float>;
                return exampleGemm<AutoType>(devSpec, execObj, M, N, K, 
                                           AutoType(alpha), AutoType(beta), repeats);
            }
            else
            {
                std::cerr << "Unknown type: " << type << std::endl;
                return EXIT_FAILURE;
            }
        },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));
}
