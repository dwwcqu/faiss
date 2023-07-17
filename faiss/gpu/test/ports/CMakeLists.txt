# Defines `gtest_discover_tests()`.
include(GoogleTest)

# macro(faiss_gpu_test file)
#   get_filename_component(test_name ${file} NAME_WE)
#   add_executable(${test_name} ${file})
#   target_link_libraries(${test_name} PRIVATE faiss_gpu_test_helper)
#   gtest_discover_tests(${test_name})
# endmacro()

faiss_gpu_test(TestL2NormKernel.cpp)