#include <alpaka/alpaka.hpp>
#include <alpaka/tensor.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>
#include <iostream>
#include <chrono>

using namespace alpaka;

template<typename Cfg>
int runBasic(Cfg const& cfg){
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];
    auto sel = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = sel.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    std::size_t N = 1<<18; // 256K elements
    tensor::Tensor1D<float> a({N});
    tensor::Tensor1D<float> b({N});
    for(std::size_t i=0;i<N;++i){ a(i)=1.f; b(i)=2.f; }

    auto t0=std::chrono::high_resolution_clock::now();
    auto c = tensor::ops::add<float,1>(exec, device, queue, a, b);        // c = a + b
    auto d = tensor::ops::mul<float,1>(exec, device, queue, c, b);        // d = c * b
    auto e = tensor::ops::add_scalar<float,1>(exec, device, queue, d, 3.f);// e = d + 3
    tensor::ops::relu_inplace<float,1>(exec, device, queue, e);           // relu in-place
    onHost::wait(queue);
    auto t1=std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    std::cout << core::demangledName(exec) << " | dev: " << deviceSpec.getApi().getName()
              << " | pipeline ms: " << ms << " | sample e[0]=" << e(0) << " (expected  ( (1+2)*2 +3 )=9)" << '\n';
    return 0;
}

int main(){
    return executeForEachIfHasDevice([](auto const& tag){ return runBasic(tag); }, onHost::allBackends(onHost::enabledApis));
}
