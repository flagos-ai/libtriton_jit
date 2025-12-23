#pragma once

#include <cstdint>
#include <cstring>
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

template <typename T>
T get_next_multiple_of(T pos, T step) {
    return ((pos + step - 1) / step) * step;
}

inline std::string join_sig(const c10::SmallVector<std::string>& signature) {
    std::stringstream ss;
    for (size_t i = 0; i < signature.size(); i++) {
        if (i == 0) {
            ss << signature[i];
        } else {
            ss << "," << signature[i];
        }
    }
    return ss.str();
}

struct ParameterBuffer {
    c10::SmallVector<std::byte> buff_;
    size_t cursor_ = 0;
    c10::SmallVector<size_t> offsets_;

    void reserve(size_t new_cap) {
        const int ESTIMATED_BYTES_PER_ARG = 4;
        this->buff_.reserve(new_cap * ESTIMATED_BYTES_PER_ARG);
        this->offsets_.reserve(new_cap);
    }

    template <typename T>
    void push_arg(T&& v) {
        using U = std::decay_t<T>;
        static_assert(std::is_trivially_copyable_v<U>, "Non trivially copyable type");
        size_t align = alignof(U);
        size_t offset = get_next_multiple_of(this->cursor_, align);
        this->offsets_.push_back(offset);

        size_t size = sizeof(U);
        this->buff_.resize(offset + size);
        std::byte* ptr = this->buff_.data() + offset;
        std::memcpy(ptr, &v, size);

        this->cursor_ = offset + size;
    }

    c10::SmallVector<void*> get_ptrs() {
        c10::SmallVector<void*> ptrs;
        ptrs.reserve(this->offsets_.size());
        std::byte* start = this->buff_.data();
        for (const size_t off : this->offsets_) {
            ptrs.push_back(start + off);
        }
        return ptrs;
    }

    size_t size() const {
        return this->offsets_.size();
    }
};

enum struct ArgType : int8_t {
    NON_CONSTEXPR = 0,
    SPECIALIZED = 1,
    CONSTEXPR = 2,
};

struct StaticSignature {
    int num_args;
    std::vector<ArgType> arg_type;

    const ArgType& at(size_t i) const {
        return arg_type.at(i);
    }
};


struct ArgHandle {
    const StaticSignature& ssig;
    /* data pointer of Tensors;
    It is not that straight to extract data pointer from a tensor, since it is encapsulated
    by Storage. We gather data pointers here for them to live out of the loop while iterating
    over arguments.*/
    ParameterBuffer& buf;
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
            // Assumption: nullopt is always treated as constexpr,
            // even if the parameter is not marked as constexpr
            signature.push_back("nullopt");
        } else {
            if (ssig.at(idx) == ArgType::CONSTEXPR) {  // constexpr
                handle_constexpr(item);
            } else if (ssig.at(idx) == ArgType::SPECIALIZED) {  // specialized
                handle_specialized(item);
            } else {  // ArgType::NON_CONSTEXPR
                handle_non_constexpr(item);
            }
        }
        idx++;
    }

    void handle_tensor(const at::Tensor& item) {
        // Assumption: Tensor is never constexpr
        TORCH_CHECK(this->ssig.at(idx) != ArgType::CONSTEXPR);
        void* p_item = item.data_ptr();
        this->buf.push_arg(p_item);
        const char* dtype = to_triton_typename(item.scalar_type());

        const char* specialization = "";
        if (ssig.at(idx) == ArgType::SPECIALIZED) {
            specialization = spec(reinterpret_cast<std::uintptr_t>(p_item));
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
                this->buf.push_arg(item);
            }
            std::string sig_for_idx = fmt::format("{}{}", dtype, specialization);
            signature.push_back(sig_for_idx);
        } else {
            this->buf.push_arg(item);
            std::string sig_for_idx = fmt::format("{}", dtype);
            signature.push_back(sig_for_idx);
        }
    }

    template<typename T>
    void handle_non_constexpr(const T& item) {
        this->buf.push_arg(item);
        const char* dtype = triton_type<decltype(item)>::name;
        signature.push_back(dtype);
    }

    void append_global_scratch() {
        void* global_scratch = nullptr;
        this->buf.push_arg(global_scratch);
    }
};


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

    const StaticSignature& get_static_sig() const {
        return this->static_sig_;
    }

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

        // Storage for argument processing using ParameterBuffer
        ParameterBuffer buffer;
        buffer.reserve(num_args);  // this is a coarse estimation of parameter size
        c10::SmallVector<std::string> signature;
        signature.reserve(num_args);

        // Process arguments
        ArgHandle handler = {this->static_sig_, buffer, signature, 0};
        (handler.handle_arg(args), ...);

        // global scratch: introduced in triton 3.3
        handler.append_global_scratch();
        std::string full_signature = join_sig(signature);

        // Backend-specific context setup
        Backend::ensure_context();
        int device_index = Backend::get_device_index();

        // Get or compile kernel
        const TritonKernelImpl<Backend>& kernel =
            this->get_kernel(full_signature, num_warps, num_stages, device_index);

        // Launch kernel with signature (for NPU backend to parse argument types)
        c10::SmallVector<void*> ptrs = buffer.get_ptrs();
        kernel.launch_with_signature(grid_x, grid_y, grid_z, num_warps, stream,
                                     ptrs.data(), full_signature);
    }

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

        kernel.launch_with_signature(grid_x, grid_y, grid_z, num_warps, stream,
                                     args, full_signature);
    }

private:
    TritonJITFunctionImpl(std::string_view path, std::string_view name);
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
