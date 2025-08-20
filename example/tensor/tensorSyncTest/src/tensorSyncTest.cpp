#include <alpaka/alpaka.hpp>
#include <alpaka/tensor.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>
#include <iostream>
#include <stdexcept>

using namespace alpaka;

template<typename Cfg>
int runSyncTest(Cfg const& cfg){
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];
    auto sel = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = sel.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    // Create a 1D tensor with 5 elements
    tensor::Tensor1D<float> a({5});
    
    // Initialize on host
    for(int i = 0; i < 5; i++) {
        a(i) = i * 2.0f;  // [0, 2, 4, 6, 8]
    }
    
    // Ensure data is on device - this should work
    a.ensureOnDevice(device, queue);
    
    // Perform device operation - this will make device dirty and host stale
    tensor::Tensor1D<float> scalar_tensor({1});
    scalar_tensor(0) = 3.0f;
    auto b = tensor::ops::add<float,1>(exec, device, queue, a, scalar_tensor);  // [3, 5, 7, 9, 11]
    
    std::cout << "After device operation, attempting to access host data without sync...\n";
    
    // This should throw an exception because device is dirty and host is stale
    try {
        float value = b(0);  // This should detect sync error and throw
        std::cout << "ERROR: No exception thrown! Value accessed: " << value << "\n";
        return 1;  // Test failed
    } catch(const std::runtime_error& e) {
        std::cout << "SUCCESS: Caught expected sync error: " << e.what() << "\n";
    }
    
    // Now properly sync back to host
    b.toHost(device, queue);
    
    // This should work now
    try {
        float value = b(0);
        std::cout << "After sync, b[0] = " << value << " (expected: 3)\n";
        if(std::abs(value - 3.0f) > 1e-6f) {
            std::cout << "ERROR: Incorrect value after sync\n";
            return 1;
        }
    } catch(const std::runtime_error& e) {
        std::cout << "ERROR: Unexpected exception after sync: " << e.what() << "\n";
        return 1;
    }
    
    // Test 2: Modify host, then try device operation without sync
    std::cout << "\nTest 2: Modifying host then trying device operation...\n";
    b(1) = 99.0f;  // This makes host dirty
    
    // This should work but print a warning about potential stale device data
    try {
        tensor::Tensor1D<float> one_tensor({1});
        one_tensor(0) = 1.0f;
        auto c = tensor::ops::add<float,1>(exec, device, queue, b, one_tensor);
        std::cout << "Device operation completed (should have auto-synced)\n";
        
        // Check the result
        c.toHost(device, queue);
        float val0 = c(0);  // Should be 4 (3+1)
        float val1 = c(1);  // Should be 100 (99+1)
        
        std::cout << "After device op: c[0]=" << val0 << " (exp: 4), c[1]=" << val1 << " (exp: 100)\n";
        
        if(std::abs(val0 - 4.0f) > 1e-6f || std::abs(val1 - 100.0f) > 1e-6f) {
            std::cout << "ERROR: Values incorrect after auto-sync\n";
            return 1;
        }
    } catch(const std::exception& e) {
        std::cout << "Device operation failed: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "All synchronization tests passed!\n";
    return 0;
}

auto main() -> int
{
    std::cout << "Testing synchronization error detection...\n\n";
    
    return executeForEachIfHasDevice(
        [=](auto const& tag) { return runSyncTest(tag); },
        onHost::allBackends(onHost::enabledApis));
}
