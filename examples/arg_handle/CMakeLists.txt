add_custom_target(
    copy_triton_axpy_src
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/axpy.py
            ${CMAKE_CURRENT_BINARY_DIR}/axpy.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/axpy.py
)

add_library(axpy_op SHARED axpy_op.cpp)
target_include_directories(axpy_op
    PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(axpy_op
    PUBLIC Torch::Torch
    PRIVATE TritonJIT::triton_jit
)
add_dependencies(axpy_op copy_triton_axpy_src)

add_executable(test_axpy test_axpy.cpp)
target_link_libraries(test_axpy
    PRIVATE axpy_op TritonJIT::triton_jit Torch::Torch GTest::gtest GTest::gtest_main)
