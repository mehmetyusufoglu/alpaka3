/* Copyright 2025 Alpaka Tensor Library Contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <vector>
#include <memory>
#include <string>
#include <cassert>

namespace alpaka
{
    namespace tensor
    {
        // Simplified data type enumeration
        enum class DataType : unsigned char {
            Float32,
            Float16,
            Int32,
            Int8
        };

        // Memory layout enumeration
        enum class Layout : unsigned char {
            RowMajor,    // C-style: rightmost dimension is contiguous
            NCHW,        // Batch, Channels, Height, Width (NVIDIA preferred)
            NHWC,        // Batch, Height, Width, Channels (CPU/mobile preferred)
            NC,          // Batch, Channels (for fully connected layers)
        };

        template<typename T, ::std::size_t Rank>
        class Tensor {
        public:
            using Shape = alpaka::Vec<::std::size_t, Rank>;
            using value_type = T;
            static constexpr ::std::size_t rank = Rank;

        private:
            Shape shape_;
            ::std::vector<T> host_data_;
            ::alpaka::onHost::Device device_;
            DataType dtype_;
            Layout layout_;
            ::std::string name_;

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
                   ::alpaka::onHost::Device device,
                   DataType dtype = DataType::Float32,
                   Layout layout = Layout::RowMajor,
                   ::std::string name = "")
                : shape_(shape)
                , device_(device)
                , dtype_(dtype)
                , layout_(layout)
                , name_(::std::move(name)) {
                
                ::std::size_t size = calculateSize();
                host_data_.resize(size);
            }

            // Copy constructor
            Tensor(const Tensor& other)
                : shape_(other.shape_)
                , host_data_(other.host_data_)
                , device_(other.device_)
                , dtype_(other.dtype_)
                , layout_(other.layout_)
                , name_(other.name_ + "_copy") {
            }

            // Move constructor
            Tensor(Tensor&& other) noexcept
                : shape_(other.shape_)
                , host_data_(::std::move(other.host_data_))
                , device_(other.device_)
                , dtype_(other.dtype_)
                , layout_(other.layout_)
                , name_(::std::move(other.name_)) {
            }

            // Assignment operators
            Tensor& operator=(const Tensor& other) {
                if (this != &other) {
                    shape_ = other.shape_;
                    host_data_ = other.host_data_;
                    device_ = other.device_;
                    dtype_ = other.dtype_;
                    layout_ = other.layout_;
                    name_ = other.name_ + "_copy";
                }
                return *this;
            }

            Tensor& operator=(Tensor&& other) noexcept {
                if (this != &other) {
                    shape_ = other.shape_;
                    host_data_ = ::std::move(other.host_data_);
                    device_ = other.device_;
                    dtype_ = other.dtype_;
                    layout_ = other.layout_;
                    name_ = ::std::move(other.name_);
                }
                return *this;
            }

            // Data access
            T* hostData() { return host_data_.data(); }
            const T* hostData() const { return host_data_.data(); }

            // Shape and metadata access
            const Shape& shape() const { return shape_; }
            ::std::size_t size() const { return calculateSize(); }
            ::std::size_t sizeBytes() const { return calculateSize() * sizeof(T); }
            
            DataType dtype() const { return dtype_; }
            Layout layout() const { return layout_; }
            const ::std::string& name() const { return name_; }
            const ::alpaka::onHost::Device& device() const { return device_; }

            // Utility methods
            void setName(const ::std::string& name) { name_ = name; }
            
            void zero() {
                ::std::fill(host_data_.begin(), host_data_.end(), T{0});
            }

            void fill(T value) {
                ::std::fill(host_data_.begin(), host_data_.end(), value);
            }

            // Element access (host only, for debugging/testing)
            template<typename... Indices>
            T& operator()(Indices... indices) {
                static_assert(sizeof...(indices) == Rank, "Number of indices must match tensor rank");
                ::std::array<::std::size_t, Rank> idx{static_cast<::std::size_t>(indices)...};
                
                // Row-major indexing
                ::std::size_t flat_idx = 0;
                ::std::size_t stride = 1;
                for(int i = static_cast<int>(Rank) - 1; i >= 0; --i) {
                    assert(idx[i] < shape_[i] && "Index out of bounds");
                    flat_idx += idx[i] * stride;
                    stride *= shape_[i];
                }
                
                return host_data_[flat_idx];
            }

            template<typename... Indices>
            const T& operator()(Indices... indices) const {
                static_assert(sizeof...(indices) == Rank, "Number of indices must match tensor rank");
                ::std::array<::std::size_t, Rank> idx{static_cast<::std::size_t>(indices)...};
                
                // Row-major indexing
                ::std::size_t flat_idx = 0;
                ::std::size_t stride = 1;
                for(int i = static_cast<int>(Rank) - 1; i >= 0; --i) {
                    assert(idx[i] < shape_[i] && "Index out of bounds");
                    flat_idx += idx[i] * stride;
                    stride *= shape_[i];
                }
                
                return host_data_[flat_idx];
            }

            // Device memory management
            bool isDeviceAllocated() const {
                // For now, just return false as we're CPU-only
                return false;
            }

            void allocateDevice() {
                // Placeholder for device allocation
                // This would allocate memory on the GPU/device
            }

            // Shape compatibility checks
            bool isShapeCompatible(const Tensor& other) const {
                for(::std::size_t i = 0; i < Rank; ++i) {
                    if (shape_[i] != other.shape_[i]) return false;
                }
                return true;
            }

            bool isDeviceCompatible(const Tensor& other) const {
                return device_ == other.device_;
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
