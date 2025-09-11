/* Reduced Precision Transformer Feed-Forward (FFN) Block Benchmark
 * Pattern from MLPerf BERT/GPT: Y = (GELU(X * W1 + b1)) * W2 + b2
 * Shapes: X[M,H]; W1[H,4H]; W2[4H,H]; b1[4H]; b2[H]
 * FLOPs (approx): 16 * M * H^2 (bias & activation omitted in count)
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/precision.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include "FfnStage1Kernel.hpp"
#include "FfnStage2Kernel.hpp"

#include <random>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>

using namespace alpaka;

template<typename T>
void fillRnd(std::vector<T>& v, float lo = -1.f, float hi = 1.f)
{
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(lo, hi);
    for(auto& x : v) x = static_cast<T>(dist(rng));
}

template<typename T>
auto runFfn(auto const deviceSpec, auto const exec,
            std::size_t B, std::size_t S, std::size_t H,
            int repeats) -> int
{
    std::size_t M = B * S;          // collapse batch & sequence
    std::size_t fourH = 4 * H;
    using Idx2 = Vec<std::size_t,2u>;

    std::cout << "FFN config: B="<<B<<" S="<<S<<" H="<<H<<" (M="<<M<<") type="<< onHost::demangledName<T>() <<"\n";
    std::cout << "Using accelerator: " << onHost::demangledName(exec) << " for "
              << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << "\n";

    // Select device & queue
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);
    onHost::Queue queue = devAcc.makeQueue();

    // Host allocations
    std::vector<T> hostX(M * H);   fillRnd(hostX);
    std::vector<T> hostW1(H * fourH); fillRnd(hostW1);
    std::vector<T> hostW2(fourH * H); fillRnd(hostW2);
    std::vector<T> hostB1(fourH);  fillRnd(hostB1);
    std::vector<T> hostB2(H);      fillRnd(hostB2);
    std::vector<T> hostY(M * H, T(0));

    // Device buffers
    auto bufX  = onHost::allocHost<T>(Idx2{M,H});
    auto bufW1 = onHost::allocHost<T>(Idx2{H,fourH});
    auto bufW2 = onHost::allocHost<T>(Idx2{fourH,H});
    auto bufH  = onHost::allocHost<T>(Idx2{M,fourH}); // intermediate
    auto bufY  = onHost::allocHost<T>(Idx2{M,H});
    auto bufB1 = onHost::allocHost<T>(Vec<std::size_t,1u>{fourH});
    auto bufB2 = onHost::allocHost<T>(Vec<std::size_t,1u>{H});

    std::copy(hostX.begin(), hostX.end(), bufX.data());
    std::copy(hostW1.begin(), hostW1.end(), bufW1.data());
    std::copy(hostW2.begin(), hostW2.end(), bufW2.data());
    std::copy(hostB1.begin(), hostB1.end(), bufB1.data());
    std::copy(hostB2.begin(), hostB2.end(), bufB2.data());
    std::copy(hostY.begin(), hostY.end(), bufY.data());

    auto devX  = onHost::allocLike(devAcc, bufX);
    auto devW1 = onHost::allocLike(devAcc, bufW1);
    auto devW2 = onHost::allocLike(devAcc, bufW2);
    auto devH  = onHost::allocLike(devAcc, bufH);
    auto devY  = onHost::allocLike(devAcc, bufY);
    auto devB1 = onHost::allocLike(devAcc, bufB1);
    auto devB2 = onHost::allocLike(devAcc, bufB2);

    onHost::memcpy(queue, devX,  bufX);
    onHost::memcpy(queue, devW1, bufW1);
    onHost::memcpy(queue, devW2, bufW2);
    onHost::memcpy(queue, devB1, bufB1);
    onHost::memcpy(queue, devB2, bufB2);
    onHost::memcpy(queue, devY,  bufY);

    FfnStage1Kernel k1; FfnStage2Kernel k2;
    auto task1 = KernelBundle{k1, devX, devW1, devH, devB1.data(), M, H, fourH};
    auto task2 = KernelBundle{k2, devH, devW2, devY, devB2.data(), M, H, fourH};

    // Work division: simple 16x16 blocks for both stages
    Vec<std::size_t,2u> block{16u,16u};
    auto frame1 = onHost::FrameSpec{ divCeil(Idx2{fourH, M}, block), block }; // (cols, rows) ordering like gemmRP
    auto frame2 = onHost::FrameSpec{ divCeil(Idx2{H, M}, block), block };

    // Warmup
    queue.enqueue(exec, frame1, task1);
    queue.enqueue(exec, frame2, task2);
    onHost::wait(queue);

    auto t0 = std::chrono::high_resolution_clock::now();
    for(int r=0; r<repeats; ++r)
    {
        queue.enqueue(exec, frame1, task1);
        queue.enqueue(exec, frame2, task2);
    }
    onHost::wait(queue);
    auto t1 = std::chrono::high_resolution_clock::now();

    double timePer = std::chrono::duration<double>(t1 - t0).count() / repeats;
    double flops = 16.0 * static_cast<double>(M) * static_cast<double>(H) * static_cast<double>(H);
    double gflops = flops / timePer / 1e9;
    std::cout << "Time per run: " << timePer << " s\n";
    std::cout << "Effective FLOPs (est): " << gflops << " GFLOP/s\n";

    // Copy back final Y (optional)
    onHost::memcpy(queue, bufY, devY);
    onHost::wait(queue);
    std::cout << "Execution completed successfully!" << std::endl;
    return EXIT_SUCCESS;
}

void help(char* argv[])
{
    std::cerr << argv[0] << " --hidden H --batch B --seq S [--type T] [--repeats R]\n";
    std::cerr << "  T: float|double|tf32|bf16|fp16|auto" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    std::size_t H = 512; // hidden size
    std::size_t B = 8;   // batch
    std::size_t S = 128; // sequence length
    int repeats = 5;
    std::string type = "float";

    for(int i=1;i<argc;++i)
    {
        std::string a = argv[i];
        if(a=="--hidden" && i+1<argc) H = std::stoul(argv[++i]);
        else if(a=="--batch" && i+1<argc) B = std::stoul(argv[++i]);
        else if(a=="--seq" && i+1<argc) S = std::stoul(argv[++i]);
        else if(a=="--repeats" && i+1<argc) repeats = std::stoi(argv[++i]);
        else if(a=="--type" && i+1<argc) type = argv[++i];
        else if(a=="--help") { help(argv); return 0; }
        else { std::cerr << "Unknown arg: "<<a<<"\n"; help(argv); return 1; }
    }

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        {
            auto const& devSpec = backend[alpaka::object::deviceSpec];
            auto const& execObj = backend[alpaka::object::exec];
            if(type=="float") return runFfn<float>(devSpec, execObj, B,S,H,repeats);
            else if(type=="double") return runFfn<double>(devSpec, execObj, B,S,H,repeats);
            else if(type=="tf32") return runFfn<alpaka::TF32>(devSpec, execObj, B,S,H,repeats);
            else if(type=="bf16") return runFfn<alpaka::BF16>(devSpec, execObj, B,S,H,repeats);
            else if(type=="fp16") return runFfn<alpaka::FP16>(devSpec, execObj, B,S,H,repeats);
            else if(type=="auto") {
                using AutoT = alpaka::precision::optimal_t<decltype(devSpec), float>;
                return runFfn<AutoT>(devSpec, execObj, B,S,H,repeats);
            } else { std::cerr << "Unknown type: "<<type<<"\n"; return 1; }
        },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));
}
