add_executable(test_launch_hooks test_launch_hooks.cpp)
target_link_libraries(test_launch_hooks PRIVATE TritonJIT::triton_jit)
add_test(NAME launch_hooks COMMAND test_launch_hooks)

add_executable(test_tuple_args test_tuple_args.cpp)
target_link_libraries(test_tuple_args PRIVATE TritonJIT::triton_jit)

add_test(NAME tuple_args_cpp COMMAND test_tuple_args)
add_test(
    NAME tuple_signature_python
    COMMAND
        ${CMAKE_COMMAND} -E env
        TRITON_JIT_BACKEND=${TRITON_BACKEND_NAME}
        ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_standalone_tuple_signature.py
        ${PROJECT_SOURCE_DIR}/scripts/standalone_compile.py
)

add_executable(test_jit_function_arg test_jit_function_arg.cpp)
target_link_libraries(test_jit_function_arg PRIVATE TritonJIT::triton_jit)
target_compile_definitions(
    test_jit_function_arg
    PRIVATE TRITON_JIT_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
add_test(NAME jit_function_arg_cpp COMMAND test_jit_function_arg)
add_test(
    NAME jit_function_arg_python
    COMMAND
        ${CMAKE_COMMAND} -E env
        TRITON_JIT_BACKEND=${TRITON_BACKEND_NAME}
        ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_standalone_jit_function.py
        ${PROJECT_SOURCE_DIR}/scripts/standalone_compile.py
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/jit_function_module.py
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/jit_function_compile_module.py
)
