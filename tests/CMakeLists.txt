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

add_executable(test_device_ptr test_device_ptr.cpp)
target_link_libraries(test_device_ptr PRIVATE TritonJIT::triton_jit)
add_test(NAME device_ptr_cpp COMMAND test_device_ptr)

add_test(
    NAME standalone_equal_to_one
    COMMAND
        ${CMAKE_COMMAND} -E env
        TRITON_JIT_BACKEND=${TRITON_BACKEND_NAME}
        ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_standalone_equal_to_one.py
        ${PROJECT_SOURCE_DIR}/scripts/standalone_compile.py
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/equal_to_one_module.py
)

add_executable(test_tuned_config test_tuned_config.cpp)
target_link_libraries(test_tuned_config PRIVATE TritonJIT::triton_jit)
target_compile_definitions(
    test_tuned_config
    PRIVATE TRITON_JIT_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
add_test(NAME tuned_config_cpp COMMAND test_tuned_config)
add_test(
    NAME export_tuned_table_python
    COMMAND
        ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_export_tuned_table.py
        ${PROJECT_SOURCE_DIR}/scripts/export_tuned_table.py
)

add_executable(test_freeze test_freeze.cpp)
target_link_libraries(test_freeze PRIVATE TritonJIT::triton_jit)
add_test(NAME freeze_cpp COMMAND test_freeze)

add_test(
    NAME tuned_resolver_python
    COMMAND
        ${Python_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_tuned_resolver.py
        ${PROJECT_SOURCE_DIR}/scripts
)

if(TRITON_JIT_BUILD_TORCH_RESOLVER)
    add_executable(test_torch_tuned_resolver test_torch_tuned_resolver.cpp)
    target_link_libraries(test_torch_tuned_resolver PRIVATE TritonJIT::triton_jit_torch_resolver)
    target_compile_definitions(
        test_torch_tuned_resolver
        PRIVATE TRITON_JIT_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME torch_tuned_resolver_cpp COMMAND test_torch_tuned_resolver)
endif()

add_executable(test_tuned_safety test_tuned_safety.cpp)
target_link_libraries(test_tuned_safety PRIVATE TritonJIT::triton_jit)
add_test(NAME tuned_source_identity COMMAND test_tuned_safety identity)
add_test(NAME tuned_concurrent_publish COMMAND test_tuned_safety race)
set_tests_properties(tuned_source_identity tuned_concurrent_publish PROPERTIES TIMEOUT 15)
add_test(NAME tuned_tensor_safety
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_tuned_tensor_safety.py ${PROJECT_SOURCE_DIR}/scripts)
set_tests_properties(tuned_tensor_safety PROPERTIES TIMEOUT 30)
option(TRITON_JIT_TEST_CUDA "Enable autotune CUDA integration tests (requires a GPU)" OFF)
if(TRITON_JIT_TEST_CUDA)
    if(NOT TRITON_JIT_BUILD_TORCH_RESOLVER OR NOT TRITON_BACKEND_NAME STREQUAL "CUDA")
        message(FATAL_ERROR "TRITON_JIT_TEST_CUDA requires BACKEND=CUDA and TRITON_JIT_BUILD_TORCH_RESOLVER=ON")
    endif()
    pybind11_add_module(tuned_runtime_probe MODULE cuda/runtime_probe.cpp)
    target_link_libraries(tuned_runtime_probe PRIVATE TritonJIT::triton_jit_torch_resolver)
    foreach(mode IN ITEMS identity gil graph)
        add_test(NAME tuned_runtime_${mode}
            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cuda/runtime_safety.py
                $<TARGET_FILE_DIR:tuned_runtime_probe> ${CMAKE_CURRENT_SOURCE_DIR}/fixtures ${mode})
        set_tests_properties(tuned_runtime_${mode} PROPERTIES LABELS "autotune;cuda" TIMEOUT 120)
    endforeach()
    # plain @triton.autotune kernels: the resolver script alone, then through the bridge
    add_test(NAME tuned_native_python
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cuda/native_tuner.py
            ${PROJECT_SOURCE_DIR}/scripts python)
    add_test(NAME tuned_native_bridge
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cuda/native_tuner.py
            ${PROJECT_SOURCE_DIR}/scripts bridge $<TARGET_FILE_DIR:tuned_runtime_probe>)
    set_tests_properties(tuned_native_python tuned_native_bridge PROPERTIES LABELS "autotune;cuda" TIMEOUT 300)
endif()
