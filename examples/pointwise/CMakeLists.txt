add_custom_target(
    copy_triton_pointwise_src
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/add.py
            ${CMAKE_CURRENT_BINARY_DIR}/add.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/add.py
)

add_library(add_op SHARED add_op.cpp)
target_include_directories(add_op
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(add_op
    PUBLIC Torch::Torch
    PRIVATE TritonJIT::triton_jit
)
add_dependencies(add_op copy_triton_pointwise_src)

add_executable(test_add test_add.cpp)
target_link_libraries(test_add
    PRIVATE add_op TritonJIT::triton_jit Torch::Torch GTest::gtest GTest::gtest_main)
