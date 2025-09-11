#pragma once

#include <alpaka/core/common.hpp>
#include <cstddef>
#include <limits>

namespace alpaka
{
    // FP16 simplified wrapper (float-backed). Exact 16-bit storage can be added later if needed.
    struct FP16
    {
        using storage_type = float;
        using compute_type = float;
        storage_type value{};

        ALPAKA_FN_HOST_ACC constexpr FP16() = default;
        ALPAKA_FN_HOST_ACC constexpr FP16(float v) : value(v) {}
        ALPAKA_FN_HOST_ACC constexpr FP16(double v) : value(static_cast<float>(v)) {}
        ALPAKA_FN_HOST_ACC constexpr FP16(int v) : value(static_cast<float>(v)) {}

        ALPAKA_FN_HOST_ACC constexpr operator float() const { return value; }

        // Arithmetic
        ALPAKA_FN_ACC friend FP16 operator+(FP16 a, FP16 b) { return FP16{a.value + b.value}; }
        ALPAKA_FN_ACC friend FP16 operator-(FP16 a, FP16 b) { return FP16{a.value - b.value}; }
        ALPAKA_FN_ACC friend FP16 operator*(FP16 a, FP16 b) { return FP16{a.value * b.value}; }
        ALPAKA_FN_ACC friend FP16 operator/(FP16 a, FP16 b) { return FP16{a.value / b.value}; }

        // Compound
        ALPAKA_FN_ACC FP16& operator+=(FP16 other) { value += other.value; return *this; }
        ALPAKA_FN_ACC FP16& operator-=(FP16 other) { value -= other.value; return *this; }
        ALPAKA_FN_ACC FP16& operator*=(FP16 other) { value *= other.value; return *this; }
        ALPAKA_FN_ACC FP16& operator/=(FP16 other) { value /= other.value; return *this; }

        // Comparisons
        ALPAKA_FN_ACC friend bool operator==(FP16 a, FP16 b) { return a.value == b.value; }
        ALPAKA_FN_ACC friend bool operator!=(FP16 a, FP16 b) { return a.value != b.value; }
        ALPAKA_FN_ACC friend bool operator<(FP16 a, FP16 b) { return a.value < b.value; }
        ALPAKA_FN_ACC friend bool operator>(FP16 a, FP16 b) { return a.value > b.value; }
        ALPAKA_FN_ACC friend bool operator<=(FP16 a, FP16 b) { return a.value <= b.value; }
        ALPAKA_FN_ACC friend bool operator>=(FP16 a, FP16 b) { return a.value >= b.value; }
    };
}

// numeric_limits specialization for FP16 intentionally omitted for now to avoid early-use
// ordering issues across the codebase. We'll add it once include order is stabilized.
