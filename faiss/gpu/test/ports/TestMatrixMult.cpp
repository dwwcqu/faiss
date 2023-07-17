#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <gtest/gtest.h>
#include <cmath>
#include <faiss/gpu/utils/CopyUtils.h>
#include <faiss/gpu/utils/Transpose.h>
#include <faiss/gpu/utils/MatrixMult.h>
#include <sstream>
#include <vector>

constexpr float lossError = 0.0001f;

void runMatrixMultByCPU(std::vector<float>& c, std::vector<float>& a, std::vector<float>& b, int m, int k, int n){
    // matrix size must match
    FAISS_ASSERT(c.size() == m * n);
    FAISS_ASSERT(a.size() == m * k);
    FAISS_ASSERT(b.size() == n * k);

    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            float sums = 0.0f;
            for(int z = 0; z < k; ++z){
                sums += a[i * k + z] * b[j * k + z];
            }
            c[i * n + j] = sums;
        }
    }
}

void testMatrixMult(int m, int k, int n) {
    using namespace faiss::gpu;

    int device = randVal(0, getNumDevices() - 1);

    StandardGpuResources res;
    res.noTempMemory();
    int dim = randVal(k, k + 10);
    int numVecs = randVal(n, n + 10);
    int numQueries = randVal(m, m + 10);

    // Input data for CPU
    std::vector<float> vecs = randVecs(numVecs, dim);
    std::vector<float> queries = randVecs(numQueries, dim);
    std::vector<float> refCPU(numQueries * numVecs, 0.0f);
    runMatrixMultByCPU(refCPU, queries, vecs, numQueries, dim, numVecs);
    std::vector<float> resGPU(numQueries * numVecs, 0.0f);


    DeviceScope scope(device);
    auto stream = res.getDefaultStream(device);

    // Copy input data to GPU
    auto gpuVecs = toDeviceNonTemporary<float, 2>(
            res.getResources().get(),
            device,
            vecs.data(),
            stream,
            {numVecs, dim});
    auto gpuQueries = toDeviceNonTemporary<float, 2>(
            res.getResources().get(),
            device,
            queries.data(),
            stream,
            {numQueries, dim});

    DeviceTensor<float, 2, true> gpuResults(
            res.getResources().get(),
            makeDevAlloc(AllocType::Other, stream),
            {numQueries, numVecs});

    runMatrixMult<float, float>(
        gpuResults, false, 
        gpuQueries, false, 
        gpuVecs, true, 
        1, 0, 
        res.getResources()->getBlasHandleCurrentDevice(),
        stream);

    fromDevice(gpuResults, resGPU.data(), stream);
    for(int i = 0; i < numQueries; ++i){
        for(int j = 0; j < numVecs; ++j){
            ASSERT_FLOAT_EQ(refCPU[i * numVecs + j], resGPU[i * numVecs + j]) << "[" << i + 1 << ", " << j + 1 << "] not equal";
        }
    }
}

TEST(TestMatrixMult, SmallMatrix) {
    testMatrixMult(20, 20, 100);
}

TEST(TestMatrixMult, BigMatrix) {
    testMatrixMult(100, 50, 3000);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
