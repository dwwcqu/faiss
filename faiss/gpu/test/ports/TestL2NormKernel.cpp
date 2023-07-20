/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <gtest/gtest.h>
#include <cmath>
#include <faiss/gpu/utils/CopyUtils.h>
#include <faiss/gpu/utils/Transpose.h>
#include <faiss/gpu/impl/L2Norm.h>
#include <sstream>
#include <vector>

constexpr float lossError = 0.0001f;
std::vector<float> computeCPUL2Norm(std::vector<float>& vecs, int numVecs, int dim, bool inputRowMajor, bool normSquard){
    FAISS_ASSERT(vecs.size() == numVecs * dim);
    std::vector<float> res(numVecs, -1.f);
    for(int i = 0; i < numVecs; ++i){
        float sum = 0.0f;
        for(int j = 0; j < dim; ++j){
            sum += vecs[i * dim + j] * vecs[i * dim + j];
        }
        res[i] = normSquard ? sum : std::sqrt(sum);
    }
    return res;
}

void testL2Norm(bool inputRowMajor, bool normSquard) {
    using namespace faiss::gpu;

    int device = randVal(0, getNumDevices() - 1);

    StandardGpuResources res;
    res.noTempMemory();

    int dim = randVal(20, 100);
    int numVecs = randVal(1000, 30000);

    // Input data for CPU
    std::vector<float> vecs = randVecs(numVecs, dim);

    std::vector<float> refNorm = computeCPUL2Norm(vecs, numVecs, dim, inputRowMajor, normSquard);

    DeviceScope scope(device);
    auto stream = res.getDefaultStream(device);

    // Copy input data to GPU
    auto gpuVecs = toDeviceNonTemporary<float, 2>(
            res.getResources().get(),
            device,
            vecs.data(),
            stream,
            {numVecs, dim});
    DeviceTensor<float, 2, true> vecsT(
            res.getResources().get(),
            makeDevAlloc(AllocType::Other, stream),
            {dim, numVecs});
    runTransposeAny(gpuVecs, 0, 1, vecsT, stream);
    // the transposed vecs
    std::vector<float> transpose(numVecs * dim, -1.f);
    fromDevice(vecsT, transpose.data(), stream);
    // determine the transpose is correct
    for(int i = 0; i < numVecs; ++i){
        for(int j = 0; j < dim; ++j){
            ASSERT_EQ(vecs[i * dim + j], transpose[j * numVecs + i]);
        }
    }

    DeviceTensor<float, 1, true> outNorms(
            res.getResources().get(),
            makeDevAlloc(AllocType::Other, stream),
            {numVecs});
    std::vector<float> gpuNorms(numVecs, -1.f);
    runL2Norm(inputRowMajor ? gpuVecs : vecsT, inputRowMajor, outNorms, normSquard, stream);

    fromDevice(outNorms.data(), gpuNorms.data(), numVecs, stream);
    for(int i = 0; i < numVecs; ++i){
        float sub = std::abs(refNorm[i] - gpuNorms[i]);
        ASSERT_LE(sub, lossError);
    }
}

TEST(TestL2NormKernel, RowMajorNormSquard) {
    testL2Norm(true, true);
}

TEST(TestL2NormKernel, NRowMajorNormSquard) {
    testL2Norm(false, true);
}

TEST(TestL2NormKernel, RowMajorNNormSquard) {
    testL2Norm(true, false);
}

TEST(TestL2NormKernel, NRowMajorNNormSquard) {
    testL2Norm(false, false);
}
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
