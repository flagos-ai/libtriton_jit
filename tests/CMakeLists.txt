# Test executables for libtriton_jit

# Test C++20 Concepts support
add_executable(test_concepts test_concepts.cpp)
target_compile_features(test_concepts PRIVATE cxx_std_20)
set_target_properties(test_concepts PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Add test to CTest
enable_testing()
add_test(NAME ConceptsTest COMMAND test_concepts)
