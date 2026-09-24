# ==============================================================================
# PPU (T-Head XuanTie GPU) Backend Configuration
# ==============================================================================

message(STATUS "Configuring PPU backend...")

# ------------------------------- PPU SDK Detection ---------------------------
# Find the PPU SDK (PPU_SDK/PPU_HOME) which provides hggc.h and libhggc.so
# (HGGC runtime), plus the ppu-llc / llvm-irformatter tools used by the
# FlagTree ppu backend at compile time.
if(NOT DEFINED PPU_HOME)
    set(PPU_HOME "/usr/local/PPU_SDK")
endif()

message(STATUS "PPU_HOME: ${PPU_HOME}")

# Find HGGC runtime library
find_library(PPU_RUNTIME_LIB hggc
    PATHS "${PPU_HOME}/lib64" "${PPU_HOME}/lib"
    NO_DEFAULT_PATH
)

# If not found, try system paths
if(NOT PPU_RUNTIME_LIB)
    find_library(PPU_RUNTIME_LIB hggc)
endif()

# Find PPU include directory
find_path(PPU_INCLUDE_DIR hggc.h
    PATHS "${PPU_HOME}/include"
    NO_DEFAULT_PATH
)

if(PPU_RUNTIME_LIB AND PPU_INCLUDE_DIR)
    message(STATUS "Found PPU Runtime: ${PPU_RUNTIME_LIB}")
    message(STATUS "Found PPU Include: ${PPU_INCLUDE_DIR}")

    # Create PPU::ppu_runtime imported target
    # GLOBAL so that consumers (e.g. FlagFFT) added via add_subdirectory
    # can link against it from the parent scope.
    if(NOT TARGET PPU::ppu_runtime)
        add_library(PPU::ppu_runtime INTERFACE IMPORTED GLOBAL)
        set_target_properties(PPU::ppu_runtime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PPU_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${PPU_RUNTIME_LIB}"
        )
        message(STATUS "Created PPU::ppu_runtime imported target")
    endif()
else()
    message(FATAL_ERROR "PPU SDK not found at ${PPU_HOME}. Please set PPU_HOME to the PPU SDK installation.")
endif()

# ------------------------------- torch PPU Runtime Check --------------------
# NOTE: The PPU backend masquerades as a CUDA device in PyTorch (torch.cuda),
# so no extra runtime registration import (like torch_musa on MUSA) is needed.

# ------------------------------- Helper Function ------------------------------
function(target_link_ppu_libraries target_name)
    message(STATUS "Linking target ${target_name} with PPU libraries")
    target_link_libraries(${target_name} PRIVATE PPU::ppu_runtime)
endfunction()

message(STATUS "PPU backend configuration complete")
