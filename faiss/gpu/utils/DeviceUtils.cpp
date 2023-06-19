/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <hip/hip_profile.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <mutex>
#include <unordered_map>

namespace faiss {
namespace gpu {

int getCurrentDevice() {
    int dev = -1;
    CUDA_VERIFY(hipGetDevice(&dev));
    FAISS_ASSERT(dev != -1);

    return dev;
}

void setCurrentDevice(int device) {
    CUDA_VERIFY(hipSetDevice(device));
}

int getNumDevices() {
    int numDev = -1;
    hipError_t err = hipGetDeviceCount(&numDev);
    if (hipErrorNoDevice == err) {
        numDev = 0;
    } else {
        CUDA_VERIFY(err);
    }
    FAISS_ASSERT(numDev != -1);

    return numDev;
}

void profilerStart() {
    CUDA_VERIFY(hipProfilerStart());
}

void profilerStop() {
    CUDA_VERIFY(hipProfilerStop());
}

void synchronizeAllDevices() {
    for (int i = 0; i < getNumDevices(); ++i) {
        DeviceScope scope(i);

        CUDA_VERIFY(hipDeviceSynchronize());
    }
}

const hipDeviceProp_t& getDeviceProperties(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, hipDeviceProp_t> properties;

    std::lock_guard<std::mutex> guard(mutex);

    auto it = properties.find(device);
    if (it == properties.end()) {
        hipDeviceProp_t prop;
        CUDA_VERIFY(hipGetDeviceProperties(&prop, device));

        properties[device] = prop;
        it = properties.find(device);
    }

    return it->second;
}

const hipDeviceProp_t& getCurrentDeviceProperties() {
    return getDeviceProperties(getCurrentDevice());
}

int getMaxThreads(int device) {
    return getDeviceProperties(device).maxThreadsPerBlock;
}

int getMaxThreadsCurrentDevice() {
    return getMaxThreads(getCurrentDevice());
}

size_t getMaxSharedMemPerBlock(int device) {
    return getDeviceProperties(device).sharedMemPerBlock;
}

size_t getMaxSharedMemPerBlockCurrentDevice() {
    return getMaxSharedMemPerBlock(getCurrentDevice());
}

int getDeviceForAddress(const void* p) {
    if (!p) {
        return -1;
    }

    hipPointerAttribute_t att;
    hipError_t err = hipPointerGetAttributes(&att, p);
    FAISS_ASSERT_FMT(
            err == hipSuccess || err == hipErrorInvalidValue,
            "unknown error %d",
            (int)err);

    if (err == hipErrorInvalidValue) {
        // Make sure the current thread error status has been reset
        err = hipGetLastError();
        FAISS_ASSERT_FMT(
                err == hipErrorInvalidValue, "unknown error %d", (int)err);
        return -1;
    }
#ifdef __HIP_PLATFORM_NVCC__
    // memoryType is deprecated for CUDA 10.0+
#if CUDA_VERSION < 10000
    if (att.memoryType == hipMemoryTypeHost) {
        return -1;
    } else {
        return att.device;
    }
#else
    // FIXME: what to use for managed memory?
    if (att.type == hipMemoryTypeDevice) {
        return att.device;
    } else {
        return -1;
    }
#endif
#else
    if (att.type == hipMemoryTypeDevice) {
        return att.device;
    } else {
        return -1;
    }
#endif
}

bool getFullUnifiedMemSupport(int device) {
    const auto& prop = getDeviceProperties(device);
    return (prop.major >= 6);
}

bool getFullUnifiedMemSupportCurrentDevice() {
    return getFullUnifiedMemSupport(getCurrentDevice());
}

bool getTensorCoreSupport(int device) {
    const auto& prop = getDeviceProperties(device);
    return (prop.major >= 7);
}

bool getTensorCoreSupportCurrentDevice() {
    return getTensorCoreSupport(getCurrentDevice());
}

size_t getFreeMemory(int device) {
    DeviceScope scope(device);

    size_t free = 0;
    size_t total = 0;

    CUDA_VERIFY(hipMemGetInfo(&free, &total));

    return free;
}

size_t getFreeMemoryCurrentDevice() {
    size_t free = 0;
    size_t total = 0;

    CUDA_VERIFY(hipMemGetInfo(&free, &total));

    return free;
}

DeviceScope::DeviceScope(int device) {
    if (device >= 0) {
        int curDevice = getCurrentDevice();

        if (curDevice != device) {
            prevDevice_ = curDevice;
            setCurrentDevice(device);
            return;
        }
    }

    // Otherwise, we keep the current device
    prevDevice_ = -1;
}

DeviceScope::~DeviceScope() {
    if (prevDevice_ != -1) {
        setCurrentDevice(prevDevice_);
    }
}

CublasHandleScope::CublasHandleScope() {
    auto blasStatus = hipblasCreate(&blasHandle_);
    FAISS_ASSERT(blasStatus == HIPBLAS_STATUS_SUCCESS);
}

CublasHandleScope::~CublasHandleScope() {
    auto blasStatus = hipblasDestroy(blasHandle_);
    FAISS_ASSERT(blasStatus == HIPBLAS_STATUS_SUCCESS);
}

CudaEvent::CudaEvent(hipStream_t stream, bool timer) : event_(0) {
    CUDA_VERIFY(hipEventCreateWithFlags(
            &event_, timer ? hipEventDefault : hipEventDisableTiming));
    CUDA_VERIFY(hipEventRecord(event_, stream));
}

CudaEvent::CudaEvent(CudaEvent&& event) noexcept
        : event_(std::move(event.event_)) {
    event.event_ = 0;
}

CudaEvent::~CudaEvent() {
    if (event_) {
        CUDA_VERIFY(hipEventDestroy(event_));
    }
}

CudaEvent& CudaEvent::operator=(CudaEvent&& event) noexcept {
    event_ = std::move(event.event_);
    event.event_ = 0;

    return *this;
}

void CudaEvent::streamWaitOnEvent(hipStream_t stream) {
    CUDA_VERIFY(hipStreamWaitEvent(stream, event_, 0));
}

void CudaEvent::cpuWaitOnEvent() {
    CUDA_VERIFY(hipEventSynchronize(event_));
}

} // namespace gpu
} // namespace faiss
