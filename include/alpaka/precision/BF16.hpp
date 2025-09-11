#pragma once

#include <alpaka/core/common.hpp>
// Standard headers to ensure std::size_t and integer types are declared
#include <cstddef>
#include <cstdint>
#include <limits>

namespace alpaka
{
    // Brain Float 16 wrapper: stores as float for portability; compute as float
    struct BF16
    {
        using storage_type = float;
        using compute_type = float;

        storage_type value{};

        ALPAKA_FN_HOST_ACC constexpr BF16() = default;
        ALPAKA_FN_HOST_ACC constexpr BF16(float v) : value(v) {}
        ALPAKA_FN_HOST_ACC constexpr BF16(double v) : value(static_cast<float>(v)) {}
        ALPAKA_FN_HOST_ACC constexpr BF16(int v) : value(static_cast<float>(v)) {}

        ALPAKA_FN_HOST_ACC constexpr operator float() const { return value; }

        // Arithmetic
        ALPAKA_FN_ACC friend BF16 operator+(BF16 a, BF16 b) { return BF16{a.value + b.value}; }
        ALPAKA_FN_ACC friend BF16 operator-(BF16 a, BF16 b) { return BF16{a.value - b.value}; }
        ALPAKA_FN_ACC friend BF16 operator*(BF16 a, BF16 b) { return BF16{a.value * b.value}; }
        ALPAKA_FN_ACC friend BF16 operator/(BF16 a, BF16 b) { return BF16{a.value / b.value}; }

        // Compound
        ALPAKA_FN_ACC BF16& operator+=(BF16 other) { value += other.value; return *this; }
        ALPAKA_FN_ACC BF16& operator-=(BF16 other) { value -= other.value; return *this; }
        ALPAKA_FN_ACC BF16& operator*=(BF16 other) { value *= other.value; return *this; }
        ALPAKA_FN_ACC BF16& operator/=(BF16 other) { value /= other.value; return *this; }

        // Comparisons
        ALPAKA_FN_ACC friend bool operator==(BF16 a, BF16 b) { return a.value == b.value; }
        ALPAKA_FN_ACC friend bool operator!=(BF16 a, BF16 b) { return a.value != b.value; }
        ALPAKA_FN_ACC friend bool operator<(BF16 a, BF16 b) { return a.value < b.value; }
        ALPAKA_FN_ACC friend bool operator>(BF16 a, BF16 b) { return a.value > b.value; }
        ALPAKA_FN_ACC friend bool operator<=(BF16 a, BF16 b) { return a.value <= b.value; }
        ALPAKA_FN_ACC friend bool operator>=(BF16 a, BF16 b) { return a.value >= b.value; }
    };
}
