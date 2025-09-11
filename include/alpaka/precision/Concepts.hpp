#pragma once

#include <type_traits>
#include <concepts>
#include <cstddef>

namespace alpaka::precision
{
    // Minimal concept for reduced precision numeric-like types
    template<typename T>
    concept ReducedNumeric = requires(T a, T b) {
        typename T::storage_type;
        typename T::compute_type;
        { T{0.0f} };
        { static_cast<float>(a) } -> ::std::convertible_to<float>;
        { a + b } -> ::std::same_as<T>;
        { a - b } -> ::std::same_as<T>;
        { a * b } -> ::std::same_as<T>;
        { a / b } -> ::std::same_as<T>;
        { a == b } -> ::std::convertible_to<bool>;
    };
}
