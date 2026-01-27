/**
 * @file benchmark_utils.h
 * @brief Performance measurement utilities for Triton JIT operators
 */

#ifndef TRITON_JIT_BENCHMARK_UTILS_H
#define TRITON_JIT_BENCHMARK_UTILS_H

#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

namespace triton_jit {
namespace benchmark {

/**
 * @brief Statistics from a benchmark run
 */
struct BenchmarkStats {
    double mean;
    double std_dev;
    double min_val;
    double max_val;
    double median;
    double p90;  // 90th percentile
    double p99;  // 99th percentile
    int num_samples;
};

/**
 * @brief High-resolution timer for benchmarking
 */
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    void start() {
        start_time_ = Clock::now();
    }

    void stop() {
        end_time_ = Clock::now();
    }

    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(end_time_ - start_time_).count();
    }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
    }

    double elapsed_s() const {
        return std::chrono::duration<double>(end_time_ - start_time_).count();
    }

private:
    TimePoint start_time_;
    TimePoint end_time_;
};

/**
 * @brief Calculate statistics from a vector of measurements
 */
inline BenchmarkStats calculate_stats(std::vector<double>& samples) {
    BenchmarkStats stats;
    stats.num_samples = static_cast<int>(samples.size());

    if (samples.empty()) {
        return stats;
    }

    // Sort for percentiles
    std::sort(samples.begin(), samples.end());

    // Min/Max
    stats.min_val = samples.front();
    stats.max_val = samples.back();

    // Mean
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.mean = sum / samples.size();

    // Standard deviation
    double sq_sum = 0;
    for (double val : samples) {
        sq_sum += (val - stats.mean) * (val - stats.mean);
    }
    stats.std_dev = std::sqrt(sq_sum / samples.size());

    // Median
    size_t mid = samples.size() / 2;
    if (samples.size() % 2 == 0) {
        stats.median = (samples[mid - 1] + samples[mid]) / 2;
    } else {
        stats.median = samples[mid];
    }

    // Percentiles
    size_t p90_idx = static_cast<size_t>(0.90 * samples.size());
    size_t p99_idx = static_cast<size_t>(0.99 * samples.size());
    stats.p90 = samples[std::min(p90_idx, samples.size() - 1)];
    stats.p99 = samples[std::min(p99_idx, samples.size() - 1)];

    return stats;
}

/**
 * @brief Calculate memory bandwidth in GB/s
 * @param bytes_accessed Total bytes read + written
 * @param latency_us Latency in microseconds
 */
inline double calculate_bandwidth_gbps(int64_t bytes_accessed, double latency_us) {
    if (latency_us <= 0) return 0;
    return (bytes_accessed / 1e9) / (latency_us / 1e6);
}

/**
 * @brief Calculate TFLOPS
 * @param flop_count Total floating-point operations
 * @param latency_us Latency in microseconds
 */
inline double calculate_tflops(int64_t flop_count, double latency_us) {
    if (latency_us <= 0) return 0;
    return (flop_count / 1e12) / (latency_us / 1e6);
}

/**
 * @brief Calculate GFLOPS
 * @param flop_count Total floating-point operations
 * @param latency_us Latency in microseconds
 */
inline double calculate_gflops(int64_t flop_count, double latency_us) {
    if (latency_us <= 0) return 0;
    return (flop_count / 1e9) / (latency_us / 1e6);
}

/**
 * @brief Helper to calculate bytes for common tensor operations
 */
struct BytesCalculator {
    /**
     * @brief Bytes for element-wise binary operation (read 2 inputs, write 1 output)
     */
    static int64_t elementwise_binary(int64_t numel, int64_t element_size) {
        return numel * element_size * 3;  // 2 reads + 1 write
    }

    /**
     * @brief Bytes for element-wise unary operation
     */
    static int64_t elementwise_unary(int64_t numel, int64_t element_size) {
        return numel * element_size * 2;  // 1 read + 1 write
    }

    /**
     * @brief Bytes for reduction operation
     */
    static int64_t reduction(int64_t input_numel, int64_t output_numel, int64_t element_size) {
        return (input_numel + output_numel) * element_size;
    }

    /**
     * @brief Bytes for matrix multiplication (M, K) x (K, N) -> (M, N)
     */
    static int64_t matmul(int64_t M, int64_t K, int64_t N, int64_t element_size) {
        return (M * K + K * N + M * N) * element_size;
    }
};

/**
 * @brief Helper to calculate FLOPs for common operations
 */
struct FlopsCalculator {
    /**
     * @brief FLOPs for element-wise operation
     */
    static int64_t elementwise(int64_t numel, int ops_per_element = 1) {
        return numel * ops_per_element;
    }

    /**
     * @brief FLOPs for reduction (sum, mean)
     */
    static int64_t reduction(int64_t numel) {
        return numel;  // One add per element
    }

    /**
     * @brief FLOPs for softmax
     */
    static int64_t softmax(int64_t batch_size, int64_t dim_size) {
        // exp: 1 FLOP each, sum: dim_size FLOPs, div: 1 FLOP each
        return batch_size * (dim_size * 3 + dim_size);
    }

    /**
     * @brief FLOPs for matrix multiplication
     */
    static int64_t matmul(int64_t M, int64_t K, int64_t N) {
        return 2 * M * K * N;  // multiply and add
    }

    /**
     * @brief FLOPs for batched matrix multiplication
     */
    static int64_t bmm(int64_t batch, int64_t M, int64_t K, int64_t N) {
        return batch * matmul(M, K, N);
    }
};

/**
 * @brief Benchmark runner with configurable warmup and iterations
 */
class BenchmarkRunner {
public:
    BenchmarkRunner(int warmup_iters = 10, int bench_iters = 100)
        : warmup_iters_(warmup_iters), bench_iters_(bench_iters) {}

    void set_warmup_iters(int iters) { warmup_iters_ = iters; }
    void set_bench_iters(int iters) { bench_iters_ = iters; }

    /**
     * @brief Run a benchmark with a synchronization callback
     * @param op_func The operation to benchmark
     * @param sync_func Function to synchronize the device
     * @return Statistics from the benchmark
     */
    template<typename OpFunc, typename SyncFunc>
    BenchmarkStats run(OpFunc&& op_func, SyncFunc&& sync_func) {
        // Warmup
        for (int i = 0; i < warmup_iters_; ++i) {
            op_func();
        }
        sync_func();

        // Benchmark
        std::vector<double> latencies;
        latencies.reserve(bench_iters_);

        for (int i = 0; i < bench_iters_; ++i) {
            timer_.start();
            op_func();
            sync_func();
            timer_.stop();
            latencies.push_back(timer_.elapsed_us());
        }

        return calculate_stats(latencies);
    }

private:
    int warmup_iters_;
    int bench_iters_;
    Timer timer_;
};

/**
 * @brief Compare performance between two implementations
 */
struct PerformanceComparison {
    double baseline_latency_us;
    double tested_latency_us;
    double speedup;
    std::string baseline_name;
    std::string tested_name;

    void calculate_speedup() {
        if (tested_latency_us > 0) {
            speedup = baseline_latency_us / tested_latency_us;
        } else {
            speedup = 0;
        }
    }
};

} // namespace benchmark
} // namespace triton_jit

#endif // TRITON_JIT_BENCHMARK_UTILS_H
