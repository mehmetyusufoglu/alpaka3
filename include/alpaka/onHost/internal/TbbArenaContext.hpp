/* SPDX-License-Identifier: MPL-2.0 */

#pragma once

#if ALPAKA_TBB

#    include <oneapi/tbb/info.h>
#    include <oneapi/tbb/task_arena.h>

#    include <cstddef>

namespace alpaka::onHost::internal
{
    /** Lightweight wrapper owning a oneTBB task_arena for queue-scoped work isolation.
     *
     *  A task_arena manages a pool of worker threads and controls where parallel tasks execute,
     *  providing thread isolation, concurrency control, and NUMA awareness. Without explicit arenas,
     *  all TBB tasks share the global pool, leading to thread oversubscription and cache pollution.
     *
     *  Each alpaka queue holds its own arena to prevent cross-queue interference, control per-queue
     *  concurrency, and improve cache locality for queue-specific kernels.
     *
     *  @see https://spec.oneapi.io/versions/latest/elements/oneTBB/source/task_scheduler/scheduling_controls/task_arena_cls.html
     */
    class TbbArenaContext
    {
    public:
        TbbArenaContext()
        {
            m_arena.initialize(oneapi::tbb::task_arena::automatic, oneapi::tbb::task_arena::automatic);
        }

        /** Execute a functor within this arena's task scheduling context.
         *
         *  @param fn Callable to execute (typically a kernel invocation lambda)
         *  @return Result of fn() invocation
         */
        template<typename TFn>
        auto execute(TFn&& fn)
        {
            return m_arena.execute(std::forward<TFn>(fn));
        }

        /** Get maximum concurrency supported by this arena.
         *
         *  @return Number of worker threads available, or default_concurrency() if unlimited
         */
        [[nodiscard]] std::size_t maxConcurrency() const
        {
            auto const concurrency = m_arena.max_concurrency();
            return concurrency != 0u ? static_cast<std::size_t>(concurrency)
                                     : static_cast<std::size_t>(oneapi::tbb::info::default_concurrency());
        }

    private:
        oneapi::tbb::task_arena m_arena;
    };
} // namespace alpaka::onHost::internal

#endif
