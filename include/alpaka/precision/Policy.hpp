#pragma once

#include <type_traits>

namespace alpaka::precision
{
    // Backend capability tags (forward-declared, user can specialize elsewhere if needed)
    template<typename TAcc>
    struct SupportsTF32 : std::false_type {};

    template<typename TAcc>
    struct SupportsBF16 : std::false_type {};

    // Select optimal reduced precision type for a given accelerator
    template<typename TAcc, typename DefaultT = float>
    struct Optimal
    {
    using type = std::conditional_t<SupportsTF32<TAcc>::value, ::alpaka::TF32,
             std::conditional_t<SupportsBF16<TAcc>::value, ::alpaka::BF16,
             std::conditional_t<std::is_same_v<DefaultT, void>, ::alpaka::FP16, DefaultT>>>;
    };

    template<typename TAcc, typename DefaultT = float>
    using optimal_t = typename Optimal<TAcc, DefaultT>::type;
}
