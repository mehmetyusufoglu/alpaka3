/* Core multi-backend Tensor implementation (host + optional device)
 * Consolidated from previous experimental versions.
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <vector>
#include <memory>
#include <string>
#include <array>
#include <cassert>
#include <any>
#include <optional>
#include <typeinfo>
#include <stdexcept>

namespace alpaka
{
    namespace tensor
    {
        enum class DataType { Float32, Float64, Int32, Int64 };
        enum class Layout { RowMajor, ColumnMajor };

        template<typename T, ::std::size_t Rank>
        class Tensor {
        public:
            using value_type = T;
            using Shape = ::std::array<::std::size_t, Rank>;
            static constexpr ::std::size_t rank = Rank;

        private:
            Shape shape_;
            using HostView = decltype(::alpaka::onHost::allocHost<T>(::alpaka::Vec<std::size_t, 1u>{1}));
            ::std::optional<HostView> host_view_;
            DataType dtype_;
            Layout layout_;
            ::std::string name_;
            ::std::any device_buffer_;
            bool hostDirty_{true};
            bool deviceDirty_{false};
            const ::std::type_info* deviceTypeInfo_{nullptr};

            ::std::size_t calculateSize() const { ::std::size_t s=1; for(::std::size_t i=0;i<Rank;++i) s*=shape_[i]; return s; }

        public:
            Tensor(Shape shape, DataType dtype = DataType::Float32, Layout layout = Layout::RowMajor, ::std::string name = "")
                : shape_(shape), dtype_(dtype), layout_(layout), name_(::std::move(name)) {
                host_view_.emplace(::alpaka::onHost::allocHost<T>(::alpaka::Vec<std::size_t,1u>{calculateSize()})); }

            Tensor(const Tensor& o)
                : shape_(o.shape_), host_view_(::std::nullopt), dtype_(o.dtype_), layout_(o.layout_), name_(o.name_+"_copy") {
                host_view_.emplace(::alpaka::onHost::allocHost<T>(::alpaka::Vec<std::size_t,1u>{o.size()}));
                for(::std::size_t i=0;i<o.size();++i) (*host_view_)[i]=(*o.host_view_)[i]; hostDirty_=true; deviceDirty_=false; }

            Tensor& operator=(const Tensor& o){ if(this!=&o){ shape_=o.shape_; if(!host_view_.has_value()||size()!=o.size()) host_view_.emplace(::alpaka::onHost::allocHost<T>(::alpaka::Vec<std::size_t,1u>{o.size()})); for(::std::size_t i=0;i<o.size();++i)(*host_view_)[i]=(*o.host_view_)[i]; dtype_=o.dtype_; layout_=o.layout_; name_=o.name_+"_assigned"; device_buffer_.reset(); hostDirty_=true; deviceDirty_=false;} return *this; }

            Tensor(Tensor&& o) noexcept
                : shape_(o.shape_), host_view_(::std::move(o.host_view_)), dtype_(o.dtype_), layout_(o.layout_), name_(::std::move(o.name_)), device_buffer_(::std::move(o.device_buffer_)), hostDirty_(o.hostDirty_), deviceDirty_(o.deviceDirty_) { o.device_buffer_.reset(); o.hostDirty_=true; o.deviceDirty_=false; }

            Tensor& operator=(Tensor&& o) noexcept { if(this!=&o){ shape_=o.shape_; host_view_=::std::move(o.host_view_); dtype_=o.dtype_; layout_=o.layout_; name_=::std::move(o.name_); device_buffer_=::std::move(o.device_buffer_); hostDirty_=o.hostDirty_; deviceDirty_=o.deviceDirty_; o.device_buffer_.reset(); o.hostDirty_=true; o.deviceDirty_=false;} return *this; }

            const Shape& shape() const { return shape_; }
            ::std::size_t size() const { return calculateSize(); }
            ::std::size_t sizeBytes() const { return size()*sizeof(T); }
            const ::std::string& name() const { return name_; }
            DataType dtype() const { return dtype_; }
            Layout layout() const { return layout_; }
            T* hostData(){ return host_view_->data(); }
            const T* hostData() const { return host_view_->data(); }

            bool isDeviceAllocated() const { return device_buffer_.has_value(); }

            template<typename Device>
            void allocateDevice(Device& device){ if(!device_buffer_.has_value()){ auto buf=::alpaka::onHost::alloc<T>(device, ::alpaka::Vec<std::size_t,1u>{size()}); device_buffer_=::std::move(buf); deviceTypeInfo_=&typeid(Device);} else if(deviceTypeInfo_ && *deviceTypeInfo_!=typeid(Device)) throw ::std::runtime_error("Tensor device buffer already allocated with different device type"); }

            void deallocateDevice(){ device_buffer_.reset(); }

            template<typename Device, typename Queue>
            void toDevice(Device& device, Queue& queue){ if(isDeviceAllocated() && hostDirty_){ auto& devBuf=getDeviceBuffer(device); ::alpaka::onHost::memcpy(queue, devBuf, *host_view_); hostDirty_=deviceDirty_=false; }}

            template<typename Device, typename Queue>
            void toHost(Device& device, Queue& queue){ if(isDeviceAllocated() && deviceDirty_){ auto& devBuf=getDeviceBuffer(device); ::alpaka::onHost::memcpy(queue, *host_view_, devBuf); deviceDirty_=hostDirty_=false; }}

            template<typename Device>
            auto& getDeviceBuffer(Device& device){ if(!device_buffer_.has_value()) allocateDevice(device); if(deviceTypeInfo_ && *deviceTypeInfo_!=typeid(Device)) throw ::std::runtime_error("Device type mismatch in getDeviceBuffer"); using BufType=decltype(::alpaka::onHost::alloc<T>(device, ::alpaka::Vec<std::size_t,1u>{1})); return *::std::any_cast<BufType>(&device_buffer_); }

            template<typename Device>
            const auto& getDeviceBuffer(Device& device) const { assert(device_buffer_.has_value() && "Device buffer not allocated"); if(deviceTypeInfo_ && *deviceTypeInfo_!=typeid(Device)) throw ::std::runtime_error("Device type mismatch in getDeviceBuffer (const)"); using BufType=decltype(::alpaka::onHost::alloc<T>(device, ::alpaka::Vec<std::size_t,1u>{1})); return *::std::any_cast<const BufType>(&device_buffer_); }

            [[deprecated("Use allocateDevice(device) with an explicit device")]] void allocateDevice() = delete;

            template<typename Device, typename Queue>
            void ensureOnDevice(Device& device, Queue& queue){ allocateDevice(device); toDevice(device, queue); }

            T& operator()(::std::size_t idx){ assert(Rank==1 && idx < size()); hostDirty_=true; return (*host_view_)[idx]; }
            const T& operator()(::std::size_t idx) const { assert(Rank==1 && idx < size()); return (*host_view_)[idx]; }

            template<typename Device, typename Queue>
            void zero(Device& device, Queue& queue){ for(::std::size_t i=0;i<size();++i)(*host_view_)[i]=T{}; if(isDeviceAllocated()) toDevice(device, queue); hostDirty_=false; }
            void zero(){ for(::std::size_t i=0;i<size();++i)(*host_view_)[i]=T{}; hostDirty_=true; }
            template<typename Device, typename Queue>
            void fill(const T& v, Device& device, Queue& queue){ for(::std::size_t i=0;i<size();++i)(*host_view_)[i]=v; if(isDeviceAllocated()) toDevice(device, queue); hostDirty_=false; }
            void fill(const T& v){ for(::std::size_t i=0;i<size();++i)(*host_view_)[i]=v; hostDirty_=true; }

            template<typename... Indices>
            T& at(Indices... indices){ static_assert(sizeof...(indices)==Rank, "Index count mismatch"); ::std::array<::std::size_t,Rank> idx{static_cast<::std::size_t>(indices)...}; ::std::size_t flat=0,stride=1; for(::std::size_t i=0;i<Rank;++i){ assert(idx[i]<shape_[i]); flat+=idx[i]*stride; stride*=shape_[i]; } return (*host_view_)[flat]; }

            void markDeviceModified(){ deviceDirty_=true; }
            bool isShapeCompatible(const Tensor& o) const { for(::std::size_t i=0;i<Rank;++i) if(shape_[i]!=o.shape_[i]) return false; return true; }
            bool isDeviceCompatible(const Tensor&) const { return true; }
            template<typename Queue> void wait(Queue& queue){ ::alpaka::onHost::wait(queue);}        };

        template<typename T> using Tensor1D = Tensor<T,1>; template<typename T> using Tensor2D = Tensor<T,2>;
        template<typename T> using Tensor3D = Tensor<T,3>; template<typename T> using Tensor4D = Tensor<T,4>;
        using FloatTensor1D = Tensor1D<float>; using FloatTensor2D = Tensor2D<float>;
        using FloatTensor3D = Tensor3D<float>; using FloatTensor4D = Tensor4D<float>;
    }
}
