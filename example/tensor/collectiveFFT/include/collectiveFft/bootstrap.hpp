#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace collectiveFft
{
    struct MultiProcessBootstrap
    {
        bool enabled = false;
        bool finalizeMpi = false;
        int worldRank = 0;
        int worldSize = 1;
        int localRank = -1;
        int localSize = -1;
        std::string disableReason{};
        std::vector<std::byte> uniqueId{};
    };

    MultiProcessBootstrap prepareBootstrap(int& argc, char**& argv);
    void finalizeBootstrap(MultiProcessBootstrap& bootstrap);
} // namespace collectiveFft
