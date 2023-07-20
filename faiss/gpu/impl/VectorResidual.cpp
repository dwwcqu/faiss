#include "hip/hip_runtime.h"
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#ifdef __HIP_PLATFORM_NVIDIA__
#include <math_constants.h> // in CUDA SDK, for CUDART_NAN_F
#else
#define CUDART_NAN_F (float)(0x7fffffffU)
#endif
#include <faiss/gpu/impl/VectorResidual.h>
#include <faiss/gpu/utils/ConversionOperators.h>
#include <faiss/gpu/utils/Tensor.h>

#include <algorithm>

namespace faiss {
namespace gpu {

template <typename CentroidT, bool LargeDim>
__global__ void calcResidual(
        Tensor<float, 2, true> vecs,
        Tensor<CentroidT, 2, true> centroids,
        Tensor<idx_t, 1, true> vecToCentroid,
        Tensor<float, 2, true> residuals) {
    auto vec = vecs[blockIdx.x];
    auto residual = residuals[blockIdx.x];
    auto centroidId = vecToCentroid[blockIdx.x];

    // Vector could be invalid (containing NaNs), so -1 was the
    // classified centroid
    if (centroidId == -1) {
        if (LargeDim) {
            for (idx_t i = threadIdx.x; i < vecs.getSize(1); i += blockDim.x) {
                residual[i] = CUDART_NAN_F;
            }
        } else {
            residual[threadIdx.x] = CUDART_NAN_F;
        }

        return;
    }

    auto centroid = centroids[centroidId];

    if (LargeDim) {
        for (idx_t i = threadIdx.x; i < vecs.getSize(1); i += blockDim.x) {
            residual[i] = vec[i] - ConvertTo<float>::to(centroid[i]);
        }
    } else {
        residual[threadIdx.x] =
                vec[threadIdx.x] - ConvertTo<float>::to(centroid[threadIdx.x]);
    }
}

template <typename CentroidT>
void calcResidual(
        Tensor<float, 2, true>& vecs,
        Tensor<CentroidT, 2, true>& centroids,
        Tensor<idx_t, 1, true>& vecToCentroid,
        Tensor<float, 2, true>& residuals,
        hipStream_t stream) {
    FAISS_ASSERT(vecs.getSize(1) == centroids.getSize(1));
    FAISS_ASSERT(vecs.getSize(1) == residuals.getSize(1));
    FAISS_ASSERT(vecs.getSize(0) == vecToCentroid.getSize(0));
    FAISS_ASSERT(vecs.getSize(0) == residuals.getSize(0));

    dim3 grid(vecs.getSize(0));

    idx_t maxThreads = getMaxThreadsCurrentDevice();
    bool largeDim = vecs.getSize(1) > maxThreads;
    dim3 block(std::min(vecs.getSize(1), maxThreads));

    if (largeDim) {
<<<<<<< HEAD:faiss/gpu/impl/VectorResidual.cu
        calcResidual<CentroidT, true><<<grid, block, 0, stream>>>(
                vecs, centroids, vecToCentroid, residuals);
    } else {
        calcResidual<CentroidT, false><<<grid, block, 0, stream>>>(
=======
        hipLaunchKernelGGL(HIP_KERNEL_NAME(calcResidual<IndexT, CentroidT, true>), grid, block, 0, stream, 
                vecs, centroids, vecToCentroid, residuals);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(calcResidual<IndexT, CentroidT, false>), grid, block, 0, stream, 
>>>>>>> port_to_rocm/v1.7.3:faiss/gpu/impl/VectorResidual.cpp
                vecs, centroids, vecToCentroid, residuals);
    }

    CUDA_TEST_ERROR();
}

void runCalcResidual(
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& centroids,
        Tensor<idx_t, 1, true>& vecToCentroid,
        Tensor<float, 2, true>& residuals,
<<<<<<< HEAD:faiss/gpu/impl/VectorResidual.cu
        cudaStream_t stream) {
    calcResidual<float>(vecs, centroids, vecToCentroid, residuals, stream);
=======
        hipStream_t stream) {
    calcResidual<Index::idx_t, float>(
            vecs, centroids, vecToCentroid, residuals, stream);
>>>>>>> port_to_rocm/v1.7.3:faiss/gpu/impl/VectorResidual.cpp
}

void runCalcResidual(
        Tensor<float, 2, true>& vecs,
        Tensor<half, 2, true>& centroids,
        Tensor<idx_t, 1, true>& vecToCentroid,
        Tensor<float, 2, true>& residuals,
<<<<<<< HEAD:faiss/gpu/impl/VectorResidual.cu
        cudaStream_t stream) {
    calcResidual<half>(vecs, centroids, vecToCentroid, residuals, stream);
=======
        hipStream_t stream) {
    calcResidual<Index::idx_t, half>(
            vecs, centroids, vecToCentroid, residuals, stream);
>>>>>>> port_to_rocm/v1.7.3:faiss/gpu/impl/VectorResidual.cpp
}

template <typename T>
__global__ void gatherReconstructByIds(
        Tensor<idx_t, 1, true> ids,
        Tensor<T, 2, true> vecs,
        Tensor<float, 2, true> out) {
    auto id = ids[blockIdx.x];
    auto vec = vecs[id];
    auto outVec = out[blockIdx.x];

    Convert<T, float> conv;

    for (idx_t i = threadIdx.x; i < vecs.getSize(1); i += blockDim.x) {
        outVec[i] = id == idx_t(-1) ? 0.0f : conv(vec[i]);
    }
}

template <typename T>
__global__ void gatherReconstructByRange(
        idx_t start,
        idx_t num,
        Tensor<T, 2, true> vecs,
        Tensor<float, 2, true> out) {
    auto id = start + blockIdx.x;
    auto vec = vecs[id];
    auto outVec = out[blockIdx.x];

    Convert<T, float> conv;

    for (idx_t i = threadIdx.x; i < vecs.getSize(1); i += blockDim.x) {
        outVec[i] = id == idx_t(-1) ? 0.0f : conv(vec[i]);
    }
}

template <typename T>
void gatherReconstructByIds(
        Tensor<idx_t, 1, true>& ids,
        Tensor<T, 2, true>& vecs,
        Tensor<float, 2, true>& out,
        hipStream_t stream) {
    FAISS_ASSERT(ids.getSize(0) == out.getSize(0));
    FAISS_ASSERT(vecs.getSize(1) == out.getSize(1));

    dim3 grid(ids.getSize(0));

    idx_t maxThreads = getMaxThreadsCurrentDevice();
    dim3 block(std::min(vecs.getSize(1), maxThreads));

<<<<<<< HEAD:faiss/gpu/impl/VectorResidual.cu
    gatherReconstructByIds<T><<<grid, block, 0, stream>>>(ids, vecs, out);
=======
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gatherReconstructByIds<IndexT, T>), grid, block, 0, stream, ids, vecs, out);
>>>>>>> port_to_rocm/v1.7.3:faiss/gpu/impl/VectorResidual.cpp

    CUDA_TEST_ERROR();
}

template <typename T>
void gatherReconstructByRange(
        idx_t start,
        idx_t num,
        Tensor<T, 2, true>& vecs,
        Tensor<float, 2, true>& out,
        hipStream_t stream) {
    FAISS_ASSERT(num > 0);
    FAISS_ASSERT(num == out.getSize(0));
    FAISS_ASSERT(vecs.getSize(1) == out.getSize(1));
    FAISS_ASSERT(start + num <= vecs.getSize(0));

    dim3 grid(num);

    idx_t maxThreads = getMaxThreadsCurrentDevice();
    dim3 block(std::min(vecs.getSize(1), maxThreads));

<<<<<<< HEAD:faiss/gpu/impl/VectorResidual.cu
    gatherReconstructByRange<T>
            <<<grid, block, 0, stream>>>(start, num, vecs, out);
=======
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gatherReconstructByRange<IndexT, T>), grid, block, 0, stream, start, num, vecs, out);
>>>>>>> port_to_rocm/v1.7.3:faiss/gpu/impl/VectorResidual.cpp

    CUDA_TEST_ERROR();
}

void runReconstruct(
        Tensor<idx_t, 1, true>& ids,
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& out,
<<<<<<< HEAD:faiss/gpu/impl/VectorResidual.cu
        cudaStream_t stream) {
    gatherReconstructByIds<float>(ids, vecs, out, stream);
=======
        hipStream_t stream) {
    gatherReconstructByIds<Index::idx_t, float>(ids, vecs, out, stream);
>>>>>>> port_to_rocm/v1.7.3:faiss/gpu/impl/VectorResidual.cpp
}

void runReconstruct(
        Tensor<idx_t, 1, true>& ids,
        Tensor<half, 2, true>& vecs,
        Tensor<float, 2, true>& out,
<<<<<<< HEAD:faiss/gpu/impl/VectorResidual.cu
        cudaStream_t stream) {
    gatherReconstructByIds<half>(ids, vecs, out, stream);
=======
        hipStream_t stream) {
    gatherReconstructByIds<Index::idx_t, half>(ids, vecs, out, stream);
>>>>>>> port_to_rocm/v1.7.3:faiss/gpu/impl/VectorResidual.cpp
}

void runReconstruct(
        idx_t start,
        idx_t num,
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& out,
<<<<<<< HEAD:faiss/gpu/impl/VectorResidual.cu
        cudaStream_t stream) {
    gatherReconstructByRange<float>(start, num, vecs, out, stream);
=======
        hipStream_t stream) {
    gatherReconstructByRange<Index::idx_t, float>(
            start, num, vecs, out, stream);
>>>>>>> port_to_rocm/v1.7.3:faiss/gpu/impl/VectorResidual.cpp
}

void runReconstruct(
        idx_t start,
        idx_t num,
        Tensor<half, 2, true>& vecs,
        Tensor<float, 2, true>& out,
<<<<<<< HEAD:faiss/gpu/impl/VectorResidual.cu
        cudaStream_t stream) {
    gatherReconstructByRange<half>(start, num, vecs, out, stream);
=======
        hipStream_t stream) {
    gatherReconstructByRange<Index::idx_t, half>(start, num, vecs, out, stream);
>>>>>>> port_to_rocm/v1.7.3:faiss/gpu/impl/VectorResidual.cpp
}

} // namespace gpu
} // namespace faiss
