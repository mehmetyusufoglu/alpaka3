#pragma once

namespace alpaka::precision
{
    // Default alignment recommendation for reduced-precision wrappers.
    // For now, wrappers are float-backed, so align to alignof(float).
    template <typename T>
    struct alignment_of
    {
        static constexpr std::size_t value = alignof(float);
    };

    template <typename T>
    inline constexpr std::size_t alignment_of_v = alignment_of<T>::value;
}
