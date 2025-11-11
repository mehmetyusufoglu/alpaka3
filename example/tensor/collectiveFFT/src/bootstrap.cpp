#include "collectiveFft/bootstrap.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>

#ifdef ALPAKA_TENSOR_COLLECTIVE_DEMO_HAS_MPI
#    include <mpi.h>
#endif

#ifdef ALPAKA_HAS_NCCL
#    include <nccl.h>
#endif

namespace collectiveFft
{
    namespace
    {
#ifndef ALPAKA_TENSOR_COLLECTIVE_DEMO_HAS_MPI
        [[nodiscard]] std::optional<int> parseEnvInt(char const* envName)
        {
            if(char const* value = std::getenv(envName))
            {
                char* end = nullptr;
                long parsed = std::strtol(value, &end, 10);
                if(end != nullptr && end != value && *end == '\0' && parsed >= 0)
                {
                    return static_cast<int>(parsed);
                }
            }
            return std::nullopt;
        }

        template<std::size_t N>
        [[nodiscard]] std::optional<int> parseFirstEnvInt(std::array<char const*, N> const& envNames)
        {
            for(auto const* name : envNames)
            {
                if(auto parsed = parseEnvInt(name))
                    return parsed;
            }
            return std::nullopt;
        }
#endif
    } // namespace

    MultiProcessBootstrap prepareBootstrap(int& argc, char**& argv)
    {
        MultiProcessBootstrap bootstrap{};

#ifdef ALPAKA_TENSOR_COLLECTIVE_DEMO_HAS_MPI
        int initialized = 0;
        auto const initQueryStatus = MPI_Initialized(&initialized);
        if(initQueryStatus != MPI_SUCCESS)
        {
            bootstrap.disableReason = "MPI_Initialized failed with status=" + std::to_string(initQueryStatus);
            return bootstrap;
        }

        if(!initialized)
        {
            int provided = MPI_THREAD_SINGLE;
            auto const initStatus = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
            if(initStatus != MPI_SUCCESS)
            {
                bootstrap.disableReason = "MPI_Init_thread failed with status=" + std::to_string(initStatus);
                return bootstrap;
            }
            bootstrap.finalizeMpi = true;
        }

        auto const commSizeStatus = MPI_Comm_size(MPI_COMM_WORLD, &bootstrap.worldSize);
        if(commSizeStatus != MPI_SUCCESS)
        {
            bootstrap.disableReason = "MPI_Comm_size failed with status=" + std::to_string(commSizeStatus);
            bootstrap.worldSize = 1;
        }

        auto const commRankStatus = MPI_Comm_rank(MPI_COMM_WORLD, &bootstrap.worldRank);
        if(commRankStatus != MPI_SUCCESS)
        {
            bootstrap.disableReason = "MPI_Comm_rank failed with status=" + std::to_string(commRankStatus);
            bootstrap.worldRank = 0;
        }
        bootstrap.enabled = bootstrap.worldSize > 1;

        if(bootstrap.worldSize > 1)
        {
            MPI_Comm localComm = MPI_COMM_NULL;
            auto const splitStatus = MPI_Comm_split_type(
                MPI_COMM_WORLD,
                MPI_COMM_TYPE_SHARED,
                bootstrap.worldRank,
                MPI_INFO_NULL,
                &localComm);
            if(splitStatus == MPI_SUCCESS && localComm != MPI_COMM_NULL)
            {
                MPI_Comm_rank(localComm, &bootstrap.localRank);
                MPI_Comm_size(localComm, &bootstrap.localSize);
                MPI_Comm_free(&localComm);
            }
            else
            {
                bootstrap.localRank = bootstrap.worldRank;
                bootstrap.localSize = bootstrap.worldSize;
            }
        }
        else
        {
            bootstrap.localRank = 0;
            bootstrap.localSize = 1;
        }

#    ifdef ALPAKA_HAS_NCCL
        if(bootstrap.enabled)
        {
            ncclUniqueId id{};
            if(bootstrap.worldRank == 0)
            {
                auto const result = ncclGetUniqueId(&id);
                if(result != ncclSuccess)
                {
                    std::cerr << "ncclGetUniqueId failed with status=" << static_cast<int>(result)
                              << "; multi-rank NCCL disabled.\n";
                    bootstrap.enabled = false;
                    bootstrap.disableReason
                        = "ncclGetUniqueId failed with status=" + std::to_string(static_cast<int>(result));
                }
            }

            int const broadcastResult
                = MPI_Bcast(&id, static_cast<int>(sizeof(ncclUniqueId)), MPI_BYTE, 0, MPI_COMM_WORLD);
            if(broadcastResult != MPI_SUCCESS)
            {
                std::cerr << "MPI_Bcast failed while distributing NCCL unique ID; multi-rank NCCL disabled.\n";
                bootstrap.enabled = false;
                bootstrap.disableReason = "MPI_Bcast failed while distributing NCCL unique ID (status="
                                          + std::to_string(broadcastResult) + ")";
            }
            int enabledFlag = bootstrap.enabled ? 1 : 0;
            MPI_Bcast(&enabledFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            bootstrap.enabled = enabledFlag != 0;

            if(bootstrap.enabled)
            {
                bootstrap.uniqueId.resize(sizeof(ncclUniqueId));
                std::memcpy(bootstrap.uniqueId.data(), &id, sizeof(ncclUniqueId));
            }
            else
            {
                bootstrap.uniqueId.clear();
            }
        }
#    else
        if(bootstrap.enabled)
        {
            bootstrap.disableReason = "NCCL support not enabled in this build.";
            bootstrap.enabled = false;
            bootstrap.uniqueId.clear();
        }
#    endif

        if(bootstrap.worldSize > 1)
        {
            int enabledFlag = bootstrap.enabled ? 1 : 0;
            MPI_Bcast(&enabledFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            bootstrap.enabled = enabledFlag != 0;

            int reasonLength = static_cast<int>(bootstrap.disableReason.size());
            MPI_Bcast(&reasonLength, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(bootstrap.worldRank != 0)
            {
                bootstrap.disableReason.resize(static_cast<std::size_t>(reasonLength));
            }
            if(reasonLength > 0)
            {
                MPI_Bcast(bootstrap.disableReason.data(), reasonLength, MPI_CHAR, 0, MPI_COMM_WORLD);
            }
        }

        return bootstrap;
#else
        static constexpr std::array<char const*, 2> worldRankKeys{"OMPI_COMM_WORLD_RANK", "RANK"};
        static constexpr std::array<char const*, 4> localRankKeys{
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "SLURM_LOCALID",
            "LOCAL_RANK"};

        bootstrap.worldRank = parseFirstEnvInt(worldRankKeys).value_or(0);
        bootstrap.worldSize = 1;
        bootstrap.localRank = parseFirstEnvInt(localRankKeys).value_or(bootstrap.worldRank);
        bootstrap.localSize = 1;
        bootstrap.enabled = false;
        bootstrap.disableReason = "Demo built without MPI support; rebuild with MPI to enable multi-rank collectives.";
        return bootstrap;
#endif
    }

    void finalizeBootstrap(MultiProcessBootstrap& bootstrap)
    {
#ifdef ALPAKA_TENSOR_COLLECTIVE_DEMO_HAS_MPI
        if(bootstrap.finalizeMpi)
        {
            MPI_Finalize();
            bootstrap.finalizeMpi = false;
        }
#else
        static_cast<void>(bootstrap);
#endif
    }
} // namespace collectiveFft
