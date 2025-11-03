/**
 * @file test_concepts.cpp
 * @brief Test C++20 Concepts support
 *
 * This file verifies that C++20 Concepts are working correctly
 * in the build environment.
 */

#include <concepts>
#include <iostream>
#include <type_traits>

// ========== Basic Concept Tests ==========

// Test 1: Simple type constraint
template<typename T>
concept Integral = std::is_integral_v<T>;

template<Integral T>
T add(T a, T b) {
    return a + b;
}

// Test 2: Multiple requirements
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
T multiply(T a, T b) {
    return a * b;
}

// Test 3: requires expression
template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};

template<Addable T>
T sum(T a, T b, T c) {
    return a + b + c;
}

// Test 4: Type member requirement (for Backend Policy)
template<typename T>
concept HasStreamType = requires {
    typename T::StreamType;
};

struct MockBackend {
    using StreamType = int;
};

template<HasStreamType Backend>
typename Backend::StreamType get_stream() {
    return typename Backend::StreamType{};
}

// ========== Main Test Function ==========

int main() {
    std::cout << "Testing C++20 Concepts support..." << std::endl;

    // Test 1: Integral concept
    int result1 = add(5, 3);
    std::cout << "✓ Test 1 (Integral): 5 + 3 = " << result1 << std::endl;

    // Test 2: Numeric concept
    double result2 = multiply(2.5, 4.0);
    std::cout << "✓ Test 2 (Numeric): 2.5 * 4.0 = " << result2 << std::endl;

    // Test 3: Addable concept
    int result3 = sum(1, 2, 3);
    std::cout << "✓ Test 3 (Addable): 1 + 2 + 3 = " << result3 << std::endl;

    // Test 4: Type member requirement
    auto stream = get_stream<MockBackend>();
    std::cout << "✓ Test 4 (HasStreamType): MockBackend has StreamType" << std::endl;

    std::cout << "\n✅ All C++20 Concepts tests passed!" << std::endl;
    std::cout << "Ready to implement Backend Policy with Concepts." << std::endl;

    return 0;
}
