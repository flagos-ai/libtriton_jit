/**
 * @file triton_jit_function_impl.h
 * @brief Template implementation of TritonJITFunction with Backend Policy
 *
 * This file contains the template implementation of TritonJITFunctionImpl,
 * which is parameterized by a BackendPolicy.
 *
 * @version 2.0.0
 * @date 2025-11-03
 */

#pragma once

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fmt/core.h"
#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"
#include "triton_jit/triton_kernel_impl.h"

namespace triton_jit {

/**
 * @brief Argument type classification
 *
 * Describes how a runtime argument is handled:
 * - NON_CONSTEXPR: Regular runtime argument
 * - SPECIALIZED: Specialized at compile-time for specific values
 * - CONSTEXPR: Compile-time constant (affects kernel specialization)
 */
enum struct ArgType : int8_t {
    NON_CONSTEXPR = 0,
    SPECIALIZED = 1,
    CONSTEXPR = 2,
};

/**
 * @brief Static signature of a Triton JIT function
 *
 * Describes the argument handling strategy for a function,
 * determined by the @triton.jit decorator in Python.
 */
struct StaticSignature {
    int num_args;
    std::vector<ArgType> arg_type;

    const ArgType& at(size_t i) const {
        return arg_type.at(i);
    }
};

/**
 * @brief Argument handler for processing variadic arguments
 *
 * This struct processes arguments passed to the kernel, extracting:
 * - Data pointers (for tensors)
 * - Kernel arguments (for kernel launch)
 * - Signature strings (for kernel specialization)
 */
struct ArgHandle {
    const StaticSignature& ssig;
    c10::SmallVector<void*>& data_pointers;
    c10::SmallVector<void*>& kernel_args;
    c10::SmallVector<std::string>& signature;
    int idx;

    template<typename... Args>
    void handle_args(Args... args) {
        (handle_arg(args), ...);
    }

    template<typename T>
    void handle_arg(const T& item) {
        if constexpr (is_optional<decltype(item)>::value) {
            handle_optional(item);
        } else if constexpr (is_same_ignore_cvref<c10::Scalar, T>::value) {
            handle_scalar(item);
        } else {
            handle_arg_plain(item);
        }
    }

    template<typename T>
    void handle_optional(const std::optional<T>& item) {
        if (item.has_value()) {
            const T& v = item.value();
            handle_arg(v);
        } else {
            handle_arg(std::nullopt);
        }
    }

    void handle_scalar(const c10::Scalar& item) {
        TORCH_CHECK(!item.isSymbolic());
        c10::ScalarType tp = item.type();
        const void* p = item.data_ptr();
        if (tp == c10::ScalarType::Bool) {
            handle_arg_plain(*reinterpret_cast<const bool*>(p));
        } else if (tp == c10::ScalarType::Long) {
            handle_arg_plain(*reinterpret_cast<const int64_t*>(p));
        } else if (tp == c10::ScalarType::UInt64) {
            handle_arg_plain(*reinterpret_cast<const uint64_t*>(p));
        } else if (tp == c10::ScalarType::Double) {
            handle_arg_plain(*reinterpret_cast<const double*>(p));
        } else {
            throw std::runtime_error("unsupported scalar type.");
        }
    }

    template<typename T>
    void handle_arg_plain(const T& item) {
        if constexpr (is_same_ignore_cvref<at::Tensor, T>::value) {
            handle_tensor(item);
        } else if constexpr (is_same_ignore_cvref<std::nullopt_t, T>::value) {
            signature.push_back("nullopt");
        } else {
            if (ssig.at(idx) == ArgType::CONSTEXPR) {
                handle_constexpr(item);
            } else if (ssig.at(idx) == ArgType::SPECIALIZED) {
                handle_specialized(item);
            } else {
                handle_non_constexpr(item);
            }
        }
        idx++;
    }

    void handle_tensor(const at::Tensor& item) {
        TORCH_CHECK(this->ssig.at(idx) != ArgType::CONSTEXPR);
        void* p_item = item.data_ptr();
        data_pointers.push_back(p_item);
        kernel_args.push_back(&(data_pointers.back()));
        const char* dtype = to_triton_typename(item.scalar_type());

        const char* specialization = "";
        if (ssig.at(idx) == ArgType::SPECIALIZED) {
            specialization = spec(reinterpret_cast<std::uintptr_t>(data_pointers.back()));
        }
        std::string sig_for_idx = fmt::format("*{}{}", dtype, specialization);
        signature.push_back(sig_for_idx);
    }

    template<typename T>
    void handle_constexpr(const T& item) {
        signature.push_back(fmt::format("{}", item));
    }

    template<typename T>
    void handle_specialized(const T& item) {
        const char* dtype = triton_type<decltype(item)>::name;
        if constexpr (std::is_integral_v<std::remove_cv_t<std::remove_reference_t<decltype(item)>>>) {
            const char* specialization = spec(item);
            if (specialization != ":1") {
                const void* p_item = &item;
                kernel_args.push_back(const_cast<void*>(p_item));
            }
            std::string sig_for_idx = fmt::format("{}{}", dtype, specialization);
            signature.push_back(sig_for_idx);
        } else {
            const void* p_item = &item;
            kernel_args.push_back(const_cast<void*>(p_item));
            std::string sig_for_idx = fmt::format("{}", dtype);
            signature.push_back(sig_for_idx);
        }
    }

    template<typename T>
    void handle_non_constexpr(const T& item) {
        const void* p_item = &item;
        kernel_args.push_back(const_cast<void*>(p_item));
        const char* dtype = triton_type<decltype(item)>::name;
        signature.push_back(dtype);
    }
};

/**
 * @brief Template TritonJITFunction class parameterized by Backend
 *
 * This class wraps a Triton JIT function for calling from C++.
 * It handles kernel compilation, caching, and launching.
 *
 * @tparam Backend Backend policy (must satisfy BackendPolicy concept)
 */
template<BackendPolicy Backend>
class TritonJITFunctionImpl {
private:
    std::string file_path_;
    std::string function_name_;
    StaticSignature static_sig_;

    /// Cached compiled kernels (keyed by signature)
    mutable std::unordered_map<std::string, TritonKernelImpl<Backend>> overloads_;

    /// Global registry of all TritonJITFunction instances
    static std::unordered_map<std::string, std::unique_ptr<TritonJITFunctionImpl<Backend>>> functions_;

public:
    /**
     * @brief Get or create a TritonJITFunction instance
     *
     * Uses the singleton pattern to ensure only one instance per (path, name) pair.
     *
     * @param path Path to Python file containing the Triton function
     * @param name Function name
     * @return Reference to the TritonJITFunction instance
     */
    static TritonJITFunctionImpl& get_instance(std::string_view path, std::string_view name) {
        std::string key = std::string(path) + "::" + std::string(name);

        auto it = functions_.find(key);
        if (it == functions_.end()) {
            // Use new instead of make_unique since constructor is private
            auto ptr = std::unique_ptr<TritonJITFunctionImpl>(new TritonJITFunctionImpl(path, name));
            functions_.emplace(key, std::move(ptr));
        }

        return *functions_.at(key);
    }

    // Delete copy constructor and assignment
    TritonJITFunctionImpl(const TritonJITFunctionImpl&) = delete;
    TritonJITFunctionImpl& operator=(const TritonJITFunctionImpl&) = delete;

    // Default move constructor and assignment
    TritonJITFunctionImpl(TritonJITFunctionImpl&&) = default;
    TritonJITFunctionImpl& operator=(TritonJITFunctionImpl&&) = default;

    /**
     * @brief Call operator - main entry point for kernel execution
     *
     * This function:
     * 1. Processes arguments to extract data pointers and build signature
     * 2. Ensures backend context is initialized
     * 3. Gets or compiles the kernel for this signature
     * 4. Launches the kernel
     *
     * @tparam Args Variadic argument types
     * @param stream Backend-specific stream
     * @param grid_x, grid_y, grid_z Grid dimensions
     * @param num_warps Number of warps per block
     * @param num_stages Number of pipeline stages
     * @param args Kernel arguments (tensors, scalars, etc.)
     */
    template<typename... Args>
    void operator()(
        typename Backend::StreamType stream,
        unsigned int grid_x,
        unsigned int grid_y,
        unsigned int grid_z,
        unsigned int num_warps,
        unsigned int num_stages,
        Args... args
    ) const {
        const int num_args = this->static_sig_.num_args;

        // Storage for argument processing
        c10::SmallVector<void*> data_pointers;
        data_pointers.reserve(num_args);
        c10::SmallVector<void*> kernel_args;
        kernel_args.reserve(num_args);
        c10::SmallVector<std::string> signature;
        signature.reserve(num_args);

        // Process arguments
        ArgHandle handler = {this->static_sig_, data_pointers, kernel_args, signature, 0};
        (handler.handle_arg(args), ...);

        // Add global scratch (Triton 3.3+)
        void* global_scratch = nullptr;
        data_pointers.push_back(global_scratch);
        kernel_args.push_back(&(data_pointers.back()));

        // Build full signature string
        std::string full_signature;
        for (size_t i = 0; i < signature.size(); i++) {
            if (i > 0) full_signature += ",";
            full_signature += signature[i];
        }

        // Backend-specific context setup
        Backend::ensure_context();
        int device_index = Backend::get_device_index();

        // Get or compile kernel
        const TritonKernelImpl<Backend>& kernel =
            this->get_kernel(full_signature, num_warps, num_stages, device_index);

        // Launch kernel
        kernel.launch(grid_x, grid_y, grid_z, num_warps,
                     stream, kernel_args.data());
    }

    /**
     * @brief Low-level API to launch kernel with raw arguments
     *
     * @param stream Backend stream
     * @param grid_x, grid_y, grid_z Grid dimensions
     * @param num_warps Number of warps
     * @param num_stages Pipeline stages
     * @param full_signature Kernel signature string
     * @param args Raw argument pointers
     */
    void launch_with_raw_args(
        typename Backend::StreamType stream,
        unsigned int grid_x,
        unsigned int grid_y,
        unsigned int grid_z,
        unsigned int num_warps,
        unsigned int num_stages,
        std::string full_signature,
        void** args
    ) const {
        Backend::ensure_context();
        int device_index = Backend::get_device_index();

        const TritonKernelImpl<Backend>& kernel =
            this->get_kernel(full_signature, num_warps, num_stages, device_index);

        kernel.launch(grid_x, grid_y, grid_z, num_warps, stream, args);
    }

private:
    /**
     * @brief Private constructor
     *
     * Loads static signature from Python introspection.
     */
    TritonJITFunctionImpl(std::string_view path, std::string_view name);

    /**
     * @brief Get or compile a kernel for given signature
     *
     * This method caches compiled kernels. If a kernel for this signature
     * doesn't exist, it triggers compilation via the Python interpreter.
     *
     * @param signature Kernel signature string
     * @param num_warps Number of warps
     * @param num_stages Pipeline stages
     * @param device_index Device index
     * @return Reference to the TritonKernel
     */
    const TritonKernelImpl<Backend>& get_kernel(
        std::string_view signature,
        int num_warps,
        int num_stages,
        int device_index
    ) const;
};

// Initialize static member
template<BackendPolicy Backend>
std::unordered_map<std::string, std::unique_ptr<TritonJITFunctionImpl<Backend>>>
    TritonJITFunctionImpl<Backend>::functions_;

// Verify move constructibility
template<BackendPolicy Backend>
static inline constexpr bool is_triton_jit_function_move_constructible_v =
    std::is_move_constructible_v<TritonJITFunctionImpl<Backend>>;

} // namespace triton_jit
