/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/DeviceTensor.h>
#include <faiss/gpu/utils/HostTensor.h>

namespace faiss {
namespace gpu {

/// Ensure the memory at `p` is either on the given device, or copy it
/// to the device in a new temporary allocation.
template <typename T, int Dim>
DeviceTensor<T, Dim, true> toDeviceTemporary(
        GpuResources* resources,
        int dstDevice,
        T* src,
<<<<<<< HEAD:faiss/gpu/utils/CopyUtils.cuh
        cudaStream_t stream,
        std::initializer_list<idx_t> sizes) {
=======
        hipStream_t stream,
        std::initializer_list<int> sizes) {
>>>>>>> port_to_rocm/v1.7.3:faiss/gpu/utils/CopyUtils.h
    int dev = getDeviceForAddress(src);
    DeviceTensor<T, Dim, true> oldT(src, sizes);

    if (dev == dstDevice) {
        // On device we expect
        return oldT;
    } else {
        // On different device or on host
        DeviceScope scope(dstDevice);

        DeviceTensor<T, Dim, true> newT(
                resources, makeTempAlloc(AllocType::Other, stream), sizes);

        newT.copyFrom(oldT, stream);
        return newT;
    }
}

template <typename T, int Dim>
DeviceTensor<T, Dim, true> toDeviceNonTemporary(
        GpuResources* resources,
        int dstDevice,
        T* src,
<<<<<<< HEAD:faiss/gpu/utils/CopyUtils.cuh
        cudaStream_t stream,
        std::initializer_list<idx_t> sizes) {
=======
        hipStream_t stream,
        std::initializer_list<int> sizes) {
>>>>>>> port_to_rocm/v1.7.3:faiss/gpu/utils/CopyUtils.h
    int dev = getDeviceForAddress(src);
    DeviceTensor<T, Dim, true> oldT(src, sizes);

    if (dev == dstDevice) {
        // On device we expect
        return oldT;
    } else {
        // On different device or on host
        DeviceScope scope(dstDevice);

        DeviceTensor<T, Dim, true> newT(
                resources, makeDevAlloc(AllocType::Other, stream), sizes);

        newT.copyFrom(oldT, stream);
        return newT;
    }
}

template <typename T>
DeviceTensor<T, 1, true> toDeviceTemporary(
        GpuResources* resources,
        const std::vector<T>& src,
        hipStream_t stream,
        int device = -1) {
    // Uses the current device if device == -1
    DeviceScope scope(device);

    DeviceTensor<T, 1, true> out(
            resources,
            makeTempAlloc(AllocType::Other, stream),
            {(idx_t)src.size()});

    out.copyFrom(src, stream);

    return out;
}

/// Copies data to the CPU, if it is not already on the CPU
template <typename T, int Dim>
HostTensor<T, Dim, true> toHost(
        T* src,
<<<<<<< HEAD:faiss/gpu/utils/CopyUtils.cuh
        cudaStream_t stream,
        std::initializer_list<idx_t> sizes) {
=======
        hipStream_t stream,
        std::initializer_list<int> sizes) {
>>>>>>> port_to_rocm/v1.7.3:faiss/gpu/utils/CopyUtils.h
    int dev = getDeviceForAddress(src);

    if (dev == -1) {
        // Already on the CPU, just wrap in a HostTensor that doesn't own this
        // memory
        return HostTensor<T, Dim, true>(src, sizes);
    } else {
        HostTensor<T, Dim, true> out(sizes);
        Tensor<T, Dim, true> devData(src, sizes);
        out.copyFrom(devData, stream);

        return out;
    }
}

/// Copies a device array's allocation to an address, if necessary
template <typename T>
inline void fromDevice(T* src, T* dst, size_t num, hipStream_t stream) {
    // It is possible that the array already represents memory at `p`,
    // in which case no copy is needed
    if (src == dst) {
        return;
    }

    int dev = getDeviceForAddress(dst);

    if (dev == -1) {
        CUDA_VERIFY(hipMemcpyAsync(
                dst, src, num * sizeof(T), hipMemcpyDeviceToHost, stream));
    } else {
        CUDA_VERIFY(hipMemcpyAsync(
                dst, src, num * sizeof(T), hipMemcpyDeviceToDevice, stream));
    }
}

/// Copies a device array's allocation to an address, if necessary
template <typename T, int Dim>
void fromDevice(Tensor<T, Dim, true>& src, T* dst, hipStream_t stream) {
    FAISS_ASSERT(src.isContiguous());
    fromDevice(src.data(), dst, src.numElements(), stream);
}

} // namespace gpu
} // namespace faiss
