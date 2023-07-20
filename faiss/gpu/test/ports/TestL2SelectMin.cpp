#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <gtest/gtest.h>
#include <cmath>
#include <faiss/gpu/utils/CopyUtils.h>
#include <faiss/gpu/utils/Transpose.h>
#include <faiss/gpu/impl/L2Select.h>
#include <sstream>
#include <vector>
#include <map>

constexpr float lossError = 0.0001f;

void runL2SelectMinCPU(std::vector<float>& productDistances, 
                       std::vector<float>& centroidDistances,
                       std::vector<float>& distances,
                       std::vector<int>& indices,
                       int k,
                       int numQueries,
                       int numVecs)
                       {
    // size must match
    FAISS_ASSERT(productDistances.size() == numQueries * numVecs);
    FAISS_ASSERT(centroidDistances.size() == numVecs);
    FAISS_ASSERT(distances.size() == numQueries * k);
    FAISS_ASSERT(indices.size() == numQueries * k);
    FAISS_ASSERT(k <= numVecs);

    for(int i = 0; i < numQueries; ++i){
        std::multimap<float, int> sorts;
        for(int j = 0; j < numVecs; ++j){
            sorts.emplace(productDistances[i * numVecs + j] + centroidDistances[j], j);
        }
        int j = 0;
        auto ite = sorts.cbegin();
        for(; j < k; ++ite, ++j){
            distances[i * k + j] = ite->first;
            indices[i * k + j] = ite->second;
        }
    }
}

void testL2SelectMin(int q, int c, int qk) {
    using namespace faiss::gpu;

    int device = randVal(0, getNumDevices() - 1);

    StandardGpuResources res;
    res.noTempMemory();

    int numVecs = randVal(c, c + 10);
    int numQueries = randVal(q, q + 10);
    int k = randVal(qk, qk + 10);
    // Input data for CPU
    std::vector<float> productDistances = randVecs(numQueries, numVecs);
    std::vector<float> centroidDistances = randVecs(1, numVecs);

    std::vector<float> refDistances(numQueries * k, 0.0f);
    std::vector<int> refIndices(numQueries * k, -1);
    runL2SelectMinCPU(productDistances, centroidDistances, refDistances, refIndices, k, numQueries, numVecs);

    DeviceScope scope(device);
    auto stream = res.getDefaultStream(device);

    // Copy input data to GPU
    auto gpuProductDistances = toDeviceNonTemporary<float, 2>(
            res.getResources().get(),
            device,
            productDistances.data(),
            stream,
            {numQueries, numVecs});

    auto gpuCentroidDistances = toDeviceNonTemporary<float, 1>(
            res.getResources().get(),
            device,
            centroidDistances.data(),
            stream,
            {numVecs});
    DeviceTensor<float, 2, true> outDistances(
            res.getResources().get(),
            makeDevAlloc(AllocType::Other, stream),
            {numQueries, k});

    DeviceTensor<int, 2, true> outIndices(
            res.getResources().get(),
            makeDevAlloc(AllocType::Other, stream),
            {numQueries, k});
    runL2SelectMin(gpuProductDistances, gpuCentroidDistances, outDistances, outIndices, k, stream);

    std::vector<float> gpuDistances(numQueries * k, 0.0f);
    std::vector<int> gpuIndices(numQueries * k, -1);
    fromDevice(outDistances, gpuDistances.data(), stream);
    fromDevice(outIndices, gpuIndices.data(), stream);
    for(int i = 0; i < numQueries; ++i){
        for(int j = 0; j < k; ++j){
            EXPECT_FLOAT_EQ(refDistances[i * k + j], gpuDistances[i * k + j]) << "[" << i + 1 << ", " << j + 1 << "]";
            float refDis = productDistances[i * numVecs + refIndices[i * k + j]] + centroidDistances[refIndices[i * k + j]];
            if(refDis != gpuDistances[i * k + j])
                EXPECT_EQ(refIndices[i * k + j], gpuIndices[i * k + j]) << "[" << i + 1 << ", " << j + 1 << "]";
        }
    }
}

TEST(TestL2SelectMin, SmallParameters){
    testL2SelectMin(5, 10, 6);
}

TEST(TestL2SelectMin, BigParameters){
    testL2SelectMin(1000, 20000, 1000);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(1000);

    return RUN_ALL_TESTS();
}
