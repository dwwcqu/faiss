/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <chrono>

namespace faiss {
namespace gpu {

/// Utility class for timing execution of a kernel
class KernelTimer {
   public:
    /// Constructor starts the timer and adds an event into the current
    /// device stream
    KernelTimer(hipStream_t stream = 0);

    /// Destructor releases event resources
    ~KernelTimer();

    /// Adds a stop event then synchronizes on the stop event to get the
    /// actual GPU-side kernel timings for any kernels launched in the
    /// current stream. Returns the number of milliseconds elapsed.
    /// Can only be called once.
    float elapsedMilliseconds();

   private:
    hipEvent_t startEvent_;
    hipEvent_t stopEvent_;
    hipStream_t stream_;
    bool valid_;
};

/// CPU wallclock elapsed timer
class CpuTimer {
   public:
    /// Creates and starts a new timer
    CpuTimer();

    /// Returns elapsed time in milliseconds
    float elapsedMilliseconds();

   private:
    std::chrono::time_point<std::chrono::steady_clock> start_;
};

} // namespace gpu
} // namespace faiss
