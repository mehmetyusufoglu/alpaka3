#pragma once

#include <alpaka/core/common.hpp>
// Standard headers to ensure std::size_t and integer types are declared
#include <cstddef>
#include <cstdint>
#include <limits>

namespace alpaka
{
    // TensorFloat-32 wrapper: stores as float, computes as float; hardware may reduce precision (CUDA Ampere+)
    struct TF32
    {
        using storage_type = float;
        using compute_type = float;
    // metadata intentionally omitted here

        storage_type value{};

        ALPAKA_FN_HOST_ACC constexpr TF32() = default;
        ALPAKA_FN_HOST_ACC constexpr TF32(float v) : value(v) {}
    ALPAKA_FN_HOST_ACC constexpr TF32(double v) : value(static_cast<float>(v)) {}
    ALPAKA_FN_HOST_ACC constexpr TF32(int v) : value(static_cast<float>(v)) {}

        ALPAKA_FN_HOST_ACC constexpr operator float() const { return value; }

        // Arithmetic
        ALPAKA_FN_ACC friend TF32 operator+(TF32 a, TF32 b) { return TF32{a.value + b.value}; }
        ALPAKA_FN_ACC friend TF32 operator-(TF32 a, TF32 b) { return TF32{a.value - b.value}; }
        ALPAKA_FN_ACC friend TF32 operator*(TF32 a, TF32 b) { return TF32{a.value * b.value}; }
        ALPAKA_FN_ACC friend TF32 operator/(TF32 a, TF32 b) { return TF32{a.value / b.value}; }

        // Compound
        ALPAKA_FN_ACC TF32& operator+=(TF32 other) { value += other.value; return *this; }
        ALPAKA_FN_ACC TF32& operator-=(TF32 other) { value -= other.value; return *this; }
        ALPAKA_FN_ACC TF32& operator*=(TF32 other) { value *= other.value; return *this; }
        ALPAKA_FN_ACC TF32& operator/=(TF32 other) { value /= other.value; return *this; }

        // Comparisons
        ALPAKA_FN_ACC friend bool operator==(TF32 a, TF32 b) { return a.value == b.value; }
        ALPAKA_FN_ACC friend bool operator!=(TF32 a, TF32 b) { return a.value != b.value; }
        ALPAKA_FN_ACC friend bool operator<(TF32 a, TF32 b) { return a.value < b.value; }
        ALPAKA_FN_ACC friend bool operator>(TF32 a, TF32 b) { return a.value > b.value; }
        ALPAKA_FN_ACC friend bool operator<=(TF32 a, TF32 b) { return a.value <= b.value; }
        ALPAKA_FN_ACC friend bool operator>=(TF32 a, TF32 b) { return a.value >= b.value; }
    };
}
