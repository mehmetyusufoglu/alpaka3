#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <typeinfo>

using namespace alpaka;

// A vector addition kernel supporting reduced-precision types
class VectorAddKernelRP {
public:
    ALPAKA_FN_ACC auto operator()(auto const& acc,
                                  alpaka::concepts::MdSpan auto const A,
                                  alpaka::concepts::MdSpan auto const B,
                                  alpaka::concepts::MdSpan auto C,
                                  auto const& numElements) const -> void {
        using namespace alpaka;
        static_assert(ALPAKA_TYPEOF(numElements)::dim() == 1, "VectorAddKernelRP expects 1D indices");

        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        simdGrid.concurrent(
            acc,
            numElements,
            [&](auto const&, auto&& simdA, auto&& simdB, auto&& simdC) constexpr {
                simdC = simdA.load() + simdB.load();
            },
            A,
            B,
            C);
    }
};

// TData can be float, alpaka::TF32, alpaka::BF16, alpaka::FP16
template <typename TData>
auto exampleRP(auto const deviceSpec, auto const exec, size_t numElements) -> int {
    using IdxVec = Vec<std::size_t, 1u>;
    IdxVec const extent(numElements);

    using Data = TData;

    std::cout << "Number of elements: " << numElements << std::endl;
    std::cout << "Element type: " << onHost::demangledName<Data>() << std::endl;
    std::cout << "Using alpaka accelerator: " << onHost::demangledName(exec) << " for "
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);
    onHost::Queue queue = devAcc.makeQueue();

    // Allocate host buffers of Data
    auto bufHostA = onHost::allocHost<Data>(extent);
    auto bufHostB = onHost::allocHostLike(bufHostA);
    auto bufHostC = onHost::allocHostLike(bufHostA);

    // Random init in float then convert to Data
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_real_distribution<float> dist(1.f, 42.f);

    for (auto i(0u); i < extent; ++i) {
        bufHostA[i] = static_cast<Data>(dist(eng));
        bufHostB[i] = static_cast<Data>(dist(eng));
        bufHostC[i] = static_cast<Data>(0.0f);
    }

    // Allocate device buffers
    auto bufAccA = onHost::allocLike(devAcc, bufHostA);
    auto bufAccB = onHost::allocLike(devAcc, bufHostB);
    auto bufAccC = onHost::allocLike(devAcc, bufHostC);

    // Copy Host -> Device
    onHost::memcpy(queue, bufAccA, bufHostA);
    onHost::memcpy(queue, bufAccB, bufHostB);
    onHost::memcpy(queue, bufAccC, bufHostC);

    // Launch
    VectorAddKernelRP kernel;
    auto const taskKernel = KernelBundle{kernel, bufAccA, bufAccB, bufAccC, extent};

    Vec<size_t, 1u> chunkSize = 256u;
    uint32_t elementsPerWorker = getNumElemPerThread<Data>(queue);
    auto dataBlocking = onHost::FrameSpec{divCeil(extent, chunkSize * elementsPerWorker), chunkSize};

    {
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        queue.enqueue(exec, dataBlocking, taskKernel);
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for kernel execution: "
                  << std::chrono::duration<double>(endT - beginT).count() << 's' << std::endl;
    }

    // Copy back
    {
        auto beginT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufHostC, bufAccC);
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for DtoH copy: "
                  << std::chrono::duration<double>(endT - beginT).count() << 's' << std::endl;
    }

    // Validate on host in float
    int falseResults = 0;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;
    for (auto i(0u); i < extent; ++i) {
        float val = static_cast<float>(bufHostC[i]);
        float correct = static_cast<float>(bufHostA[i]) + static_cast<float>(bufHostB[i]);
        if (val != correct) {
            if (falseResults < MAX_PRINT_FALSE_RESULTS)
                std::cerr << "C[" << i << "] == " << val << " != " << correct << std::endl;
            ++falseResults;
        }
    }

    if (falseResults == 0) {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    } else {
        std::cout << "Found " << falseResults << " false results, printed no more than "
                  << MAX_PRINT_FALSE_RESULTS << "\nExecution results incorrect!" << std::endl;
        return EXIT_FAILURE;
    }
}

void help(char* argv[]) { std::cerr << argv[0] << " [-n  numElements] [-t type] [-h]\n  -t type in {float,tf32,bf16,fp16}\n" << std::endl; }

int main(int argc, char* argv[]) {
    size_t numElements = 123456;
    std::string type = "float";

    int opt;
    while ((opt = getopt(argc, argv, "hn:t:")) != -1) {
        switch (opt) {
        case 'n':
            try { numElements = std::stoul(optarg, nullptr, 0); }
            catch(...) { std::cerr << "Invalid -n value" << std::endl; return EXIT_FAILURE; }
            break;
        case 't': type = optarg; break;
        case 'h': help(argv); return EXIT_SUCCESS;
        default: help(argv); return EXIT_FAILURE;
        }
    }

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend){
            auto const& devSpec = backend[alpaka::object::deviceSpec];
            auto const& exec    = backend[alpaka::object::exec];
            if (type == "float")      return exampleRP<float>(devSpec, exec, numElements);
            else if (type == "tf32")  return exampleRP<alpaka::TF32>(devSpec, exec, numElements);
            else if (type == "bf16")  return exampleRP<alpaka::BF16>(devSpec, exec, numElements);
            else if (type == "fp16")  return exampleRP<alpaka::FP16>(devSpec, exec, numElements);
            else {
                std::cerr << "Unknown type: " << type << std::endl; return EXIT_FAILURE;
            }
        },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));
}
