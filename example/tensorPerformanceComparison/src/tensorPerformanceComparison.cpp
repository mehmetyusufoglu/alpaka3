#include <alpaka/alpaka.hpp>
#include <alpaka/tensor.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>
#include <iostream>
#include <chrono>

using namespace alpaka;

template<typename Cfg>
int runPerformanceComparison(Cfg const& cfg){
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];
    auto sel = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = sel.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    std::cout << "Backend: " << alpaka::core::demangledName(exec) << std::endl;
    std::cout << "Device: " << deviceSpec.getApi().getName() << std::endl;

    // Test parameters
    constexpr std::size_t N = 1 << 20; // 1M elements
    constexpr int numIterations = 10;

    // Create test tensors
    tensor::Tensor1D<float> a({N});
    tensor::Tensor1D<float> b({N});
    tensor::Tensor1D<float> c({N});

    // Initialize with test data
    for(std::size_t i = 0; i < N; i++) {
        a(i) = static_cast<float>(i % 100) / 100.0f;
        b(i) = static_cast<float>((i + 50) % 100) / 100.0f;
        c(i) = 1.0f;
    }

    std::cout << "Tensor size: " << N << " elements (" << (N * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;

    // ============================================================================
    // Test 1: Eager Operations (Current Approach)
    // Each operation syncs to host automatically
    // ============================================================================
    
    std::cout << "Starting eager operations test..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int iter = 0; iter < numIterations; iter++) {
        // Chain: result = relu((a + b) * 2.0 + c)
        auto temp1 = tensor::ops::add<float,1>(exec, device, queue, a, b);     // Auto-sync #1
        auto temp2 = tensor::ops::mul_scalar<float,1>(exec, device, queue, temp1, 2.0f);  // Auto-sync #2
        auto temp3 = tensor::ops::add<float,1>(exec, device, queue, temp2, c);  // Auto-sync #3
        auto result = tensor::ops::relu<float,1>(exec, device, queue, temp3);   // Auto-sync #4
        
        // Force computation (access result)
        volatile float val = result(0);  // Now safe to access (already synced)
        (void)val;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto eagerTime = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Eager Operations (4 sync points per iteration):" << std::endl;
    std::cout << "  Total time: " << eagerTime << " ms" << std::endl;
    std::cout << "  Time per iteration: " << eagerTime / numIterations << " ms" << std::endl;
    std::cout << "  Operations per second: " << (4 * numIterations * 1000.0) / eagerTime << std::endl;

    // ============================================================================
    // Test 2: Lazy Operations (New Approach)
    // Operations stay on device, single sync at end
    // ============================================================================
    
    start = std::chrono::high_resolution_clock::now();
    
    for(int iter = 0; iter < numIterations; iter++) {
        // Chain: result = relu((a + b) * 2.0 + c) - all on device
        auto temp1 = tensor::ops::addLazy<float,1>(exec, device, queue, a, b);     // No sync
        auto temp2 = tensor::ops::mul_scalarLazy<float,1>(exec, device, queue, temp1, 2.0f);  // No sync
        auto temp3 = tensor::ops::addLazy<float,1>(exec, device, queue, temp2, c);  // No sync  
        auto result = tensor::ops::reluLazy<float,1>(exec, device, queue, temp3);   // No sync
        
        // Single sync at the end
        result.toHost(device, queue);
        volatile float val = result(0);  // Access result
        (void)val;
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto lazyTime = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\nLazy Operations (1 sync point per iteration):" << std::endl;
    std::cout << "  Total time: " << lazyTime << " ms" << std::endl;
    std::cout << "  Time per iteration: " << lazyTime / numIterations << " ms" << std::endl;
    std::cout << "  Operations per second: " << (4 * numIterations * 1000.0) / lazyTime << std::endl;

    // ============================================================================
    // Performance Analysis
    // ============================================================================
    
    double speedup = eagerTime / lazyTime;
    double syncOverhead = ((eagerTime - lazyTime) / eagerTime) * 100.0;

    std::cout << "\n=== PERFORMANCE ANALYSIS ===" << std::endl;
    std::cout << "Speedup: " << speedup << "x faster" << std::endl;
    std::cout << "Sync overhead reduction: " << syncOverhead << "%" << std::endl;
    std::cout << "Bandwidth saved: " << ((3 * N * sizeof(float) * numIterations) / 1024 / 1024) 
              << " MB (avoided 3 sync transfers per iteration)" << std::endl;

    // ============================================================================
    // Test 3: Proper .view() API - Using View-Based Operations
    // ============================================================================
    
    std::cout << "\n=== DEMONSTRATING PROPER .view() API ===" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    
    for(int iter = 0; iter < numIterations; iter++) {
        // Using proper view-based operations with manual chaining
        auto view1 = a.view();
        auto view2 = tensor::ops::mul_scalar(view1, exec, device, queue, 2.0f);
        auto result = tensor::ops::relu(view2, exec, device, queue);  // Final operation auto-syncs
        
        volatile float val = result(0);  // Access result  
        (void)val;
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto viewTime = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Proper View-based API:" << std::endl;
    std::cout << "  Total time: " << viewTime << " ms" << std::endl;
    std::cout << "  Time per iteration: " << viewTime / numIterations << " ms" << std::endl;
    std::cout << "  Speedup vs eager: " << eagerTime / viewTime << "x" << std::endl;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    
    return 0;
}

auto main() -> int
{
    std::cout << "Tensor Performance Comparison: Eager vs Lazy Operations\n" << std::endl;
    
    return executeForEachIfHasDevice(
        [=](auto const& tag) { return runPerformanceComparison(tag); },
        onHost::allBackends(onHost::enabledApis));
}
