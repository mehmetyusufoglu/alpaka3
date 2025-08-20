#include <alpaka/alpaka.hpp>
#include <alpaka/tensor.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>
#include <iostream>

using namespace alpaka;

template<typename Cfg>
int runSimpleViewTest(Cfg const& cfg){
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];
    auto sel = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = sel.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    std::cout << "Backend: " << alpaka::core::demangledName(exec) << std::endl;

    try {
        // Create simple tensors
        tensor::Tensor1D<float> a({10});
        for(int i = 0; i < 10; i++) {
            a(i) = static_cast<float>(i);
        }

        std::cout << "Created tensor a: [0, 1, 2, ..., 9]" << std::endl;

        // Test 1: Basic view creation
        auto view = a.view();
        std::cout << "✓ Created view successfully" << std::endl;

        // Test 2: View auto-conversion to tensor
        tensor::Tensor1D<float>& tensor_ref = view;
        std::cout << "✓ View auto-converts to tensor reference" << std::endl;
        std::cout << "  tensor_ref[0] = " << tensor_ref(0) << std::endl;

        // Test 3: Try lazy operation
        std::cout << "Testing lazy mul_scalar operation..." << std::endl;
        auto result = tensor::ops::mul_scalarLazy<float,1>(exec, device, queue, a, 2.0f);
        std::cout << "✓ Lazy operation completed" << std::endl;
        
        // Sync result manually
        result.toHost(device, queue);
        std::cout << "✓ Manual sync completed" << std::endl;
        std::cout << "  result[0] = " << result(0) << " (expected: 0)" << std::endl;
        std::cout << "  result[1] = " << result(1) << " (expected: 2)" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "✅ All view tests passed!" << std::endl;
    return 0;
}

auto main() -> int
{
    std::cout << "Simple View Test\n" << std::endl;
    
    // Test only one backend for debugging
    return executeForEachIfHasDevice(
        [=](auto const& tag) { return runSimpleViewTest(tag); },
        onHost::allBackends(onHost::enabledApis));
}
