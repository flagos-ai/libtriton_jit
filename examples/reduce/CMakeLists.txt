add_custom_target(
    copy_triton_reduce_src
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/sum.py
            ${CMAKE_CURRENT_BINARY_DIR}/sum.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/sum.py
)

add_library(sum_op SHARED sum_op.cpp)
target_include_directories(sum_op
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(sum_op
    PRIVATE TritonJIT::triton_jit
    PUBLIC Torch::Torch
)
add_dependencies(sum_op copy_triton_reduce_src)

add_executable(test_sum test_sum.cpp)
target_link_libraries(test_sum
    PRIVATE sum_op Torch::Torch GTest::gtest GTest::gtest_main)
add_dependencies(test_sum copy_triton_reduce_src)
