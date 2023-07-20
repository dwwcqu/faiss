/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/Timer.h>
#include <faiss/impl/FaissAssert.h>
#include <chrono>

namespace faiss {
namespace gpu {

KernelTimer::KernelTimer(hipStream_t stream)
        : startEvent_(0), stopEvent_(0), stream_(stream), valid_(true) {
    CUDA_VERIFY(hipEventCreate(&startEvent_));
    CUDA_VERIFY(hipEventCreate(&stopEvent_));

    CUDA_VERIFY(hipEventRecord(startEvent_, stream_));
}

KernelTimer::~KernelTimer() {
    CUDA_VERIFY(hipEventDestroy(startEvent_));
    CUDA_VERIFY(hipEventDestroy(stopEvent_));
}

float KernelTimer::elapsedMilliseconds() {
    FAISS_ASSERT(valid_);

    CUDA_VERIFY(hipEventRecord(stopEvent_, stream_));
    CUDA_VERIFY(hipEventSynchronize(stopEvent_));

    auto time = 0.0f;
    CUDA_VERIFY(hipEventElapsedTime(&time, startEvent_, stopEvent_));
    valid_ = false;

    return time;
}

CpuTimer::CpuTimer() {
    start_ = std::chrono::steady_clock::now();
}

float CpuTimer::elapsedMilliseconds() {
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start_;

    return duration.count();
}

} // namespace gpu
} // namespace faiss
