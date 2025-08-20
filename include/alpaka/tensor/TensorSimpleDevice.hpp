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

        // Simple tensor class that avoids device template issues
        template<typename T, ::std::size_t Rank>
        class Tensor {
        public:
            using value_type = T;
            using Shape = ::std::array<::std::size_t, Rank>;
            static constexpr ::std::size_t rank = Rank;

        private:
            Shape shape_;
            ::std::vector<T> host_data_;
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
            // Constructor without device for now
            Tensor(Shape shape, 
                   DataType dtype = DataType::Float32,
                   Layout layout = Layout::RowMajor,
                   ::std::string name = "")
                : shape_(shape)
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
                , dtype_(other.dtype_)
                , layout_(other.layout_)
                , name_(other.name_ + "_copy") {
            }

            // Assignment operator
            Tensor& operator=(const Tensor& other) {
                if(this != &other) {
                    shape_ = other.shape_;
                    host_data_ = other.host_data_;
                    dtype_ = other.dtype_;
                    layout_ = other.layout_;
                    name_ = other.name_ + "_assigned";
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
            T* hostData() { return host_data_.data(); }
            const T* hostData() const { return host_data_.data(); }

            // Element access
            T& operator()(::std::size_t idx) {
                assert(Rank == 1 && "Single index access only for 1D tensors");
                assert(idx < host_data_.size() && "Index out of bounds");
                return host_data_[idx];
            }

            const T& operator()(::std::size_t idx) const {
                assert(Rank == 1 && "Single index access only for 1D tensors");
                assert(idx < host_data_.size() && "Index out of bounds");
                return host_data_[idx];
            }

            // Utility methods
            void zero() {
                ::std::fill(host_data_.begin(), host_data_.end(), T{});
            }

            void fill(const T& value) {
                ::std::fill(host_data_.begin(), host_data_.end(), value);
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
                
                return host_data_[flat_idx];
            }

            // Device memory management (placeholders)
            bool isDeviceAllocated() const {
                return false;  // CPU-only for now
            }

            void allocateDevice() {
                // Placeholder - would allocate GPU memory
            }

            // Compatibility checks
            bool isShapeCompatible(const Tensor& other) const {
                for(::std::size_t i = 0; i < Rank; ++i) {
                    if (shape_[i] != other.shape_[i]) return false;
                }
                return true;
            }

            bool isDeviceCompatible(const Tensor& other) const {
                return true;  // Always compatible for CPU-only
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
