/* Copyright 2025 Alpaka Tensor Library Contributors
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
        // Data type enum for tensor elements
        enum class DataType {
            Float32,
            Float64,
            Int32,
            Int64
        };

        // Memory layout enum
        enum class Layout {
            RowMajor,
            ColumnMajor
        };

        // GPU-accelerated tensor class with flexible device handling
        template<typename T, ::std::size_t Rank>
        class Tensor {
        public:
            using value_type = T;
            using Shape = ::std::array<::std::size_t, Rank>;
            static constexpr ::std::size_t rank = Rank;

        private:
            Shape shape_;
            // Host memory as an alpaka managed view (for memcpy compatibility across backends)
            using HostView = decltype(::alpaka::onHost::allocHost<T>(::alpaka::Vec<std::size_t, 1u>{1}));
            ::std::optional<HostView> host_view_;
            DataType dtype_;
            Layout layout_;
            ::std::string name_;
            
            // Device buffer - use optional to handle allocation state
            // Type-erased device buffer (ManagedView of arbitrary API) stored in-place
            ::std::any device_buffer_;
            bool hostDirty_{true};
            bool deviceDirty_{false};
            const ::std::type_info* deviceTypeInfo_{nullptr};

            ::std::size_t calculateSize() const {
                ::std::size_t size = 1;
                for(::std::size_t i = 0; i < Rank; ++i) {
                    size *= shape_[i];
                }
                return size;
            }

        public:
            // Constructor
            Tensor(Shape shape, 
                   DataType dtype = DataType::Float32,
                   Layout layout = Layout::RowMajor,
                   ::std::string name = "")
                : shape_(shape)
                , dtype_(dtype)
                , layout_(layout)
                , name_(::std::move(name)) {
                
                ::std::size_t size = calculateSize();
                host_view_.emplace(::alpaka::onHost::allocHost<T>(::alpaka::Vec<std::size_t, 1u>{size}));
            }

            // Copy constructor
            Tensor(const Tensor& other)
                : shape_(other.shape_)
                , host_view_(::std::nullopt)
                , dtype_(other.dtype_)
                , layout_(other.layout_)
                , name_(other.name_ + "_copy") {
                host_view_.emplace(::alpaka::onHost::allocHost<T>(::alpaka::Vec<std::size_t,1u>{other.size()}));
                for(::std::size_t i=0;i<other.size();++i) (*host_view_)[i]=(*other.host_view_)[i];
                // Don't copy device buffer - require explicit allocation/ensureOnDevice
                hostDirty_ = true;
                deviceDirty_ = false;
            }

            // Assignment operator
            Tensor& operator=(const Tensor& other) {
                if(this != &other) {
                    shape_ = other.shape_;
                    // Reallocate host view if needed then copy
                    if(!host_view_.has_value() || size() != other.size()) {
                        host_view_.emplace(::alpaka::onHost::allocHost<T>(::alpaka::Vec<std::size_t,1u>{other.size()}));
                    }
                    for(::std::size_t i=0;i<other.size();++i) (*host_view_)[i]=(*other.host_view_)[i];
                    dtype_ = other.dtype_;
                    layout_ = other.layout_;
                    name_ = other.name_ + "_assigned";
                    device_buffer_.reset(); // Clear device allocation
                    hostDirty_ = true;
                    deviceDirty_ = false;
                }
                return *this;
            }

            // Move constructor
            Tensor(Tensor&& other) noexcept
                : shape_(other.shape_)
                , host_view_(::std::move(other.host_view_))
                , dtype_(other.dtype_)
                , layout_(other.layout_)
                , name_(::std::move(other.name_))
                , device_buffer_(::std::move(other.device_buffer_))
                , hostDirty_(other.hostDirty_)
                , deviceDirty_(other.deviceDirty_) {
                other.device_buffer_.reset();
                other.hostDirty_ = true;
                other.deviceDirty_ = false;
            }

            // Move assignment
            Tensor& operator=(Tensor&& other) noexcept {
                if(this != &other) {
                    shape_ = other.shape_;
                    host_view_ = ::std::move(other.host_view_);
                    dtype_ = other.dtype_;
                    layout_ = other.layout_;
                    name_ = ::std::move(other.name_);
                    device_buffer_ = ::std::move(other.device_buffer_);
                    hostDirty_ = other.hostDirty_;
                    deviceDirty_ = other.deviceDirty_;
                    other.device_buffer_.reset();
                    other.hostDirty_ = true;
                    other.deviceDirty_ = false;
                }
                return *this;
            }

            // Access methods
            const Shape& shape() const { return shape_; }
            ::std::size_t size() const { return calculateSize(); }
            ::std::size_t sizeBytes() const { return calculateSize() * sizeof(T); }
            const ::std::string& name() const { return name_; }
            
            DataType dtype() const { return dtype_; }
            Layout layout() const { return layout_; }

            // Data access
            T* hostData() { return host_view_->data(); }
            const T* hostData() const { return host_view_->data(); }

            // Device memory management - REAL GPU allocation!
            bool isDeviceAllocated() const { return device_buffer_.has_value(); }

            template<typename Device>
            void allocateDevice(Device& device) {
                if (!device_buffer_.has_value()) {
                    ::std::size_t size = calculateSize();
                    auto buf = ::alpaka::onHost::alloc<T>(device, ::alpaka::Vec<std::size_t,1u>{size});
                    device_buffer_ = ::std::move(buf);
                    deviceTypeInfo_ = &typeid(Device);
                } else {
                    // If already allocated ensure same device type (simplistic safety check)
                    if(deviceTypeInfo_ && *deviceTypeInfo_ != typeid(Device)) {
                        throw ::std::runtime_error("Tensor device buffer already allocated with different device type");
                    }
                }
            }

            void deallocateDevice() {
                device_buffer_.reset();
            }

            // Data transfer operations
        template<typename Device, typename Queue>
        void toDevice(Device& device, Queue& queue) {
                if (isDeviceAllocated() && hostDirty_) {
                    // allocate temporary view size vector
            auto& devBuf = getDeviceBuffer(device);
            ::alpaka::onHost::memcpy(queue, devBuf, *host_view_);
                    hostDirty_ = false;
                    deviceDirty_ = false;
                }
            }

        template<typename Device, typename Queue>
        void toHost(Device& device, Queue& queue) {
                if (isDeviceAllocated() && deviceDirty_) {
            auto& devBuf = getDeviceBuffer(device);
            ::alpaka::onHost::memcpy(queue, *host_view_, devBuf);
                    deviceDirty_ = false;
                    hostDirty_ = false;
                }
            }

            // Get device buffer for kernel operations
            template<typename Device>
            auto& getDeviceBuffer(Device& device) {
                if (!device_buffer_.has_value()) allocateDevice(device);
                if(deviceTypeInfo_ && *deviceTypeInfo_ != typeid(Device)) {
                    throw ::std::runtime_error("Device type mismatch in getDeviceBuffer");
                }
                using BufType = decltype(::alpaka::onHost::alloc<T>(device, ::alpaka::Vec<std::size_t,1u>{1}));
                return *::std::any_cast<BufType>(&device_buffer_);
            }

            template<typename Device>
            const auto& getDeviceBuffer(Device& device) const {
                assert(device_buffer_.has_value() && "Device buffer not allocated");
                if(deviceTypeInfo_ && *deviceTypeInfo_ != typeid(Device)) {
                    throw ::std::runtime_error("Device type mismatch in getDeviceBuffer (const)");
                }
                using BufType = decltype(::alpaka::onHost::alloc<T>(device, ::alpaka::Vec<std::size_t,1u>{1}));
                return *::std::any_cast<const BufType>(&device_buffer_);
            }

            // Removed legacy no-arg allocateDevice(): forcing caller to provide device prevents accidental host allocation.
            [[deprecated("Use allocateDevice(device) with an explicit device to avoid implicit host allocation")]]
            void allocateDevice() = delete;

            // Ensure data is present on device (alloc + upload if needed)
            template<typename Device, typename Queue>
            void ensureOnDevice(Device& device, Queue& queue) {
                allocateDevice(device);
                toDevice(device, queue);
            }

            // Element access
            T& operator()(::std::size_t idx) {
                assert(Rank == 1 && "Single index access only for 1D tensors");
                assert(idx < size() && "Index out of bounds");
                hostDirty_ = true;
                return (*host_view_)[idx];
            }

            const T& operator()(::std::size_t idx) const {
                assert(Rank == 1 && "Single index access only for 1D tensors");
                assert(idx < size() && "Index out of bounds");
                return (*host_view_)[idx];
            }

            // Utility methods
            template<typename Device, typename Queue>
            void zero(Device& device, Queue& queue) {
                for(::std::size_t i=0;i<size();++i) (*host_view_)[i]=T{};
                if (isDeviceAllocated()) {
                    toDevice(device, queue);
                }
                hostDirty_ = false; // now host & device in sync
            }

            void zero() {
                for(::std::size_t i=0;i<size();++i) (*host_view_)[i]=T{};
                // No device sync for legacy compatibility
                hostDirty_ = true;
            }

            template<typename Device, typename Queue>
            void fill(const T& value, Device& device, Queue& queue) {
                for(::std::size_t i=0;i<size();++i) (*host_view_)[i]=value;
                if (isDeviceAllocated()) {
                    toDevice(device, queue);
                }
                hostDirty_ = false;
            }

            void fill(const T& value) {
                for(::std::size_t i=0;i<size();++i) (*host_view_)[i]=value;
                // No device sync for legacy compatibility
                hostDirty_ = true;
            }

            // Multi-dimensional element access
            template<typename... Indices>
            T& at(Indices... indices) {
                static_assert(sizeof...(indices) == Rank, "Number of indices must match tensor rank");
                ::std::array<::std::size_t, Rank> idx = {static_cast<::std::size_t>(indices)...};
                
                ::std::size_t flat_idx = 0;
                ::std::size_t stride = 1;
                for(::std::size_t i = 0; i < Rank; ++i) {
                    assert(idx[i] < shape_[i] && "Index out of bounds");
                    flat_idx += idx[i] * stride;
                    stride *= shape_[i];
                }
                
                return (*host_view_)[flat_idx];
            }

            // Mark device data modified externally (e.g., after a kernel write)
            void markDeviceModified() {
                deviceDirty_ = true;
            }

            // Compatibility checks
            bool isShapeCompatible(const Tensor& other) const {
                for(::std::size_t i = 0; i < Rank; ++i) {
                    if (shape_[i] != other.shape_[i]) return false;
                }
                return true;
            }

            bool isDeviceCompatible(const Tensor& other) const {
                // Simple check - in real implementation would compare device IDs
                return true;
            }

            // Synchronization
            template<typename Queue>
            void wait(Queue& queue) {
                ::alpaka::onHost::wait(queue);
            }
        };

        // Type aliases for common tensor types
        template<typename T> using Tensor1D = Tensor<T, 1>;
        template<typename T> using Tensor2D = Tensor<T, 2>;
        template<typename T> using Tensor3D = Tensor<T, 3>;
        template<typename T> using Tensor4D = Tensor<T, 4>;

        // Common float types
        using FloatTensor1D = Tensor1D<float>;
        using FloatTensor2D = Tensor2D<float>;
        using FloatTensor3D = Tensor3D<float>;
        using FloatTensor4D = Tensor4D<float>;

    } // namespace tensor
} // namespace alpaka
