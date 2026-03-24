# ==============================================================================
# CUDA/IX Backend Configuration
# ==============================================================================

message(STATUS "Configuring CUDA backend...")

find_package(CUDAToolkit REQUIRED COMPONENTS cuda_driver)
message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")

# Helper function for CUDA targets (placeholder for consistency)
function(target_link_cuda_libraries target_name)
    target_link_libraries(${target_name} PRIVATE CUDA::cuda_driver)
endfunction()

message(STATUS "CUDA backend configuration complete")
