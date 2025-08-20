/* Copyright 2025 Alpaka Tensor Library Contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/mem/MdSpan.hpp>

#include <vector>
#include <optional>
#include <string>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <algorithm>

namespace alpaka
{
    namespace tensor
    {
        // Data type enumeration for runtime type handling
        enum class DataType : uint8_t {
            Float32,
            Float16,
            Int32,
            Int8,
            UInt8
        };

        // Memory layout enumeration for performance optimization
        enum class Layout : uint8_t {
            RowMajor,    // C-style: rightmost dimension is contiguous
            ColumnMajor, // Fortran-style: leftmost dimension is contiguous
            NCHW,        // Batch, Channels, Height, Width (NVIDIA preferred)
            NHWC,        // Batch, Height, Width, Channels (CPU/mobile preferred)
            NC,          // Batch, Channels (for fully connected layers)
            CHW,         // Channels, Height, Width (single batch)
            HWC          // Height, Width, Channels (single batch)
        };

        // Get size of data type in bytes
        constexpr ::std::size_t getDataTypeSize(DataType dtype) {
            switch(dtype) {
                case DataType::Float32: return 4;
                case DataType::Float16: return 2;
                case DataType::Int32: return 4;
                case DataType::Int8: return 1;
                case DataType::UInt8: return 1;
                default: return 4;
            }
        }

            template<typename T, ::std::size_t Rank>
    class Tensor {
    public:
        using Shape = alpaka::Vec<std::size_t, Rank>;
        using value_type = T;
        static constexpr std::size_t rank = Rank;

    private:
        // Core tensor metadata
        Shape shape_;
        DataType dtype_;
        Layout layout_;
        std::string name_;
        
        // Memory management
        std::vector<T> host_data_;
        std::optional<decltype(alpaka::onHost::alloc<T>(std::declval<alpaka::onHost::Device>(), std::size_t{}))> device_buffer_;
        alpaka::onHost::Device device_;
        bool device_allocated_ = false;

        // Helper to calculate flat size from shape
        std::size_t calculateSize() const {
            std::size_t size = 1;
            for(std::size_t i = 0; i < Rank; ++i) {
                size *= shape_[i];
            }
            return size;
        }

    public:
        // Constructor with shape, device, and optional metadata
        Tensor(Shape shape, 
               alpaka::onHost::Device device,
               DataType dtype = DataType::Float32,
               Layout layout = Layout::RowMajor,
               std::string name = "")
            : shape_(shape)
            , dtype_(dtype)
            , layout_(layout)
            , name_(std::move(name))
            , device_(device) {
            
            std::size_t size = calculateSize();
            host_data_.resize(size);
        }

        // Copy constructor (deep copy)
        Tensor(const Tensor& other)
            : shape_(other.shape_)
            , dtype_(other.dtype_)
            , layout_(other.layout_)
            , name_(other.name_ + "_copy")
            , host_data_(other.host_data_)
            , device_(other.device_)
            , device_allocated_(false) {
            // Note: device buffer not copied, will be allocated on demand
        }

        // Move constructor
        Tensor(Tensor&& other) noexcept
            : shape_(other.shape_)
            , dtype_(other.dtype_)
            , layout_(other.layout_)
            , name_(std::move(other.name_))
            , host_data_(std::move(other.host_data_))
            , device_buffer_(std::move(other.device_buffer_))
            , device_(other.device_)
            , device_allocated_(other.device_allocated_) {
            
            other.device_allocated_ = false;
        }

        // Assignment operators
        Tensor& operator=(const Tensor& other) {
            if (this != &other) {
                shape_ = other.shape_;
                dtype_ = other.dtype_;
                layout_ = other.layout_;
                name_ = other.name_ + "_copy";
                host_data_ = other.host_data_;
                device_ = other.device_;
                device_buffer_.reset();  // Force reallocation
                device_allocated_ = false;
            }
            return *this;
        }

        Tensor& operator=(Tensor&& other) noexcept {
            if (this != &other) {
                shape_ = other.shape_;
                dtype_ = other.dtype_;
                layout_ = other.layout_;
                name_ = std::move(other.name_);
                host_data_ = std::move(other.host_data_);
                device_buffer_ = std::move(other.device_buffer_);
                device_ = other.device_;
                device_allocated_ = other.device_allocated_;
                other.device_allocated_ = false;
            }
            return *this;
        }

        // Device memory management
        void allocateDevice() {
            if (!device_allocated_) {
                std::size_t size = calculateSize();
                device_buffer_ = alpaka::onHost::alloc<T>(device_, size);
                device_allocated_ = true;
            }
        }

        void deallocateDevice() {
            device_buffer_.reset();
            device_allocated_ = false;
        }

        // Memory transfers (following tutorial patterns)
        void toDevice(alpaka::onHost::Queue& queue) {
            allocateDevice();
            auto device_view = alpaka::View(*device_buffer_);
            alpaka::onHost::memcpy(queue, device_view, host_data_);
        }

        void toHost(alpaka::onHost::Queue& queue) {
            if (device_allocated_) {
                auto device_view = alpaka::View(*device_buffer_);
                alpaka::onHost::memcpy(queue, host_data_, device_view);
            }
        }

        // MdSpan access for kernels
        auto deviceMdSpan() {
            allocateDevice();
            return alpaka::onHost::makeMdSpan(*device_buffer_, shape_);
        }

        auto hostMdSpan() {
            return alpaka::onHost::makeMdSpan(host_data_.data(), shape_);
        }

        auto deviceMdSpan() const {
            assert(device_allocated_ && "Device memory not allocated");
            return alpaka::onHost::makeMdSpan(*device_buffer_, shape_);
        }

        auto hostMdSpan() const {
            return alpaka::onHost::makeMdSpan(host_data_.data(), shape_);
        }

        // Data access
        T* hostData() { return host_data_.data(); }
        const T* hostData() const { return host_data_.data(); }

        // Shape and metadata access
        const Shape& shape() const { return shape_; }
        std::size_t size() const { return calculateSize(); }
        std::size_t sizeBytes() const { return calculateSize() * sizeof(T); }
        
        DataType dtype() const { return dtype_; }
        Layout layout() const { return layout_; }
        const std::string& name() const { return name_; }
        const alpaka::onHost::Device& device() const { return device_; }
        
        bool isDeviceAllocated() const { return device_allocated_; }

        // Utility methods
        void setName(const std::string& name) { name_ = name; }
        
        void zero() {
            std::fill(host_data_.begin(), host_data_.end(), T{0});
        }

        void fill(T value) {
            std::fill(host_data_.begin(), host_data_.end(), value);
        }

        // Element access (host only, for debugging/testing)
        template<typename... Indices>
        T& operator()(Indices... indices) {
            static_assert(sizeof...(indices) == Rank, "Number of indices must match tensor rank");
            std::array<std::size_t, Rank> idx{static_cast<std::size_t>(indices)...};
            
            // Row-major indexing
            std::size_t flat_idx = 0;
            std::size_t stride = 1;
            for(int i = Rank - 1; i >= 0; --i) {
                assert(idx[i] < shape_[i] && "Index out of bounds");
                flat_idx += idx[i] * stride;
                stride *= shape_[i];
            }
            
            return host_data_[flat_idx];
        }

        template<typename... Indices>
        const T& operator()(Indices... indices) const {
            static_assert(sizeof...(indices) == Rank, "Number of indices must match tensor rank");
            std::array<std::size_t, Rank> idx{static_cast<std::size_t>(indices)...};
            
            // Row-major indexing
            std::size_t flat_idx = 0;
            std::size_t stride = 1;
            for(int i = Rank - 1; i >= 0; --i) {
                assert(idx[i] < shape_[i] && "Index out of bounds");
                flat_idx += idx[i] * stride;
                stride *= shape_[i];
            }
            
            return host_data_[flat_idx];
        }

        // Shape compatibility checks
        bool isShapeCompatible(const Tensor& other) const {
            if constexpr (Rank != other.rank) return false;
            for(std::size_t i = 0; i < Rank; ++i) {
                if (shape_[i] != other.shape_[i]) return false;
            }
            return true;
        }

        bool isDeviceCompatible(const Tensor& other) const {
            // Simple device comparison - could be more sophisticated
            return device_ == other.device_;
        }

        // Reshape (must preserve total size)
        template<std::size_t NewRank>
        Tensor<T, NewRank> reshape(alpaka::Vec<std::size_t, NewRank> new_shape) const {
            // Verify total size is preserved
            std::size_t new_size = 1;
            for(std::size_t i = 0; i < NewRank; ++i) {
                new_size *= new_shape[i];
            }
            assert(new_size == size() && "Reshape must preserve total size");

            Tensor<T, NewRank> result(new_shape, device_, dtype_, layout_, name_ + "_reshaped");
            result.host_data_ = host_data_;  // Share data
            return result;
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