/**
 * @file report_generator.h
 * @brief Report generation utilities for Triton JIT test results
 */

#ifndef TRITON_JIT_REPORT_GENERATOR_H
#define TRITON_JIT_REPORT_GENERATOR_H

#include "test_framework.h"
#include "benchmark_utils.h"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>

namespace triton_jit {
namespace report {

/**
 * @brief Test result entry for a single operator
 */
struct OperatorTestResult {
    std::string operator_name;
    std::string category;  // pointwise, reduce, matmul, etc.
    bool correctness_passed;
    test::CorrectnessResult correctness_detail;
    test::BenchmarkResult benchmark_result;
    benchmark::BenchmarkStats benchmark_stats;
    double speedup_vs_reference;  // Compared to PyTorch reference
    std::string error_message;
    std::vector<std::string> test_configs;  // List of tested configurations
};

/**
 * @brief Complete test report
 */
struct TestReport {
    std::string backend;
    std::string device_name;
    std::string timestamp;
    int total_tests;
    int passed_tests;
    int failed_tests;
    std::vector<OperatorTestResult> results;
    std::map<std::string, int> category_pass_counts;
    std::map<std::string, int> category_total_counts;
};

/**
 * @brief Report generator for various output formats
 */
class ReportGenerator {
public:
    explicit ReportGenerator(const std::string& backend_name)
        : backend_name_(backend_name) {
        // Set timestamp
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        timestamp_ = ss.str();
    }

    /**
     * @brief Add a test result
     */
    void add_result(const OperatorTestResult& result) {
        results_.push_back(result);
    }

    /**
     * @brief Generate JSON report
     */
    std::string to_json() const {
        std::stringstream ss;
        ss << "{\n";
        ss << "  \"backend\": \"" << backend_name_ << "\",\n";
        ss << "  \"timestamp\": \"" << timestamp_ << "\",\n";
        ss << "  \"summary\": {\n";

        int passed = 0, failed = 0;
        for (const auto& r : results_) {
            if (r.correctness_passed) ++passed;
            else ++failed;
        }

        ss << "    \"total_tests\": " << results_.size() << ",\n";
        ss << "    \"passed\": " << passed << ",\n";
        ss << "    \"failed\": " << failed << "\n";
        ss << "  },\n";
        ss << "  \"results\": [\n";

        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& r = results_[i];
            ss << "    {\n";
            ss << "      \"operator\": \"" << r.operator_name << "\",\n";
            ss << "      \"category\": \"" << r.category << "\",\n";
            ss << "      \"correctness\": {\n";
            ss << "        \"passed\": " << (r.correctness_passed ? "true" : "false") << ",\n";
            ss << "        \"max_abs_diff\": " << std::scientific << std::setprecision(6)
               << r.correctness_detail.max_abs_diff << ",\n";
            ss << "        \"max_rel_diff\": " << r.correctness_detail.max_rel_diff << "\n";
            ss << "      },\n";
            ss << "      \"benchmark\": {\n";
            ss << "        \"mean_latency_us\": " << std::fixed << std::setprecision(2)
               << r.benchmark_result.mean_latency_us << ",\n";
            ss << "        \"std_latency_us\": " << r.benchmark_result.std_latency_us << ",\n";
            ss << "        \"min_latency_us\": " << r.benchmark_result.min_latency_us << ",\n";
            ss << "        \"max_latency_us\": " << r.benchmark_result.max_latency_us << ",\n";
            ss << "        \"throughput_gbps\": " << r.benchmark_result.throughput_gbps << ",\n";
            ss << "        \"tflops\": " << r.benchmark_result.tflops << "\n";
            ss << "      },\n";
            ss << "      \"speedup\": " << std::setprecision(2) << r.speedup_vs_reference << "\n";
            ss << "    }";
            if (i < results_.size() - 1) ss << ",";
            ss << "\n";
        }

        ss << "  ]\n";
        ss << "}\n";

        return ss.str();
    }

    /**
     * @brief Generate CSV report
     */
    std::string to_csv() const {
        std::stringstream ss;

        // Header
        ss << "operator,category,passed,max_abs_diff,max_rel_diff,";
        ss << "mean_latency_us,std_latency_us,min_latency_us,max_latency_us,";
        ss << "throughput_gbps,tflops,speedup\n";

        for (const auto& r : results_) {
            ss << r.operator_name << ",";
            ss << r.category << ",";
            ss << (r.correctness_passed ? "true" : "false") << ",";
            ss << std::scientific << std::setprecision(6)
               << r.correctness_detail.max_abs_diff << ",";
            ss << r.correctness_detail.max_rel_diff << ",";
            ss << std::fixed << std::setprecision(2)
               << r.benchmark_result.mean_latency_us << ",";
            ss << r.benchmark_result.std_latency_us << ",";
            ss << r.benchmark_result.min_latency_us << ",";
            ss << r.benchmark_result.max_latency_us << ",";
            ss << r.benchmark_result.throughput_gbps << ",";
            ss << r.benchmark_result.tflops << ",";
            ss << r.speedup_vs_reference << "\n";
        }

        return ss.str();
    }

    /**
     * @brief Generate Markdown report
     */
    std::string to_markdown() const {
        std::stringstream ss;

        ss << "# Triton JIT Operator Test Report\n\n";
        ss << "**Backend:** " << backend_name_ << "\n\n";
        ss << "**Timestamp:** " << timestamp_ << "\n\n";

        // Summary
        int passed = 0, failed = 0;
        for (const auto& r : results_) {
            if (r.correctness_passed) ++passed;
            else ++failed;
        }

        ss << "## Summary\n\n";
        ss << "| Metric | Value |\n";
        ss << "|--------|-------|\n";
        ss << "| Total Tests | " << results_.size() << " |\n";
        ss << "| Passed | " << passed << " |\n";
        ss << "| Failed | " << failed << " |\n";
        ss << "| Pass Rate | " << std::fixed << std::setprecision(1)
           << (100.0 * passed / results_.size()) << "% |\n\n";

        // Results table
        ss << "## Detailed Results\n\n";
        ss << "| Operator | Category | Passed | Mean Latency (us) | Throughput (GB/s) | Speedup |\n";
        ss << "|----------|----------|--------|-------------------|-------------------|--------|\n";

        for (const auto& r : results_) {
            ss << "| " << r.operator_name;
            ss << " | " << r.category;
            ss << " | " << (r.correctness_passed ? ":white_check_mark:" : ":x:");
            ss << " | " << std::fixed << std::setprecision(2) << r.benchmark_result.mean_latency_us;
            ss << " | " << r.benchmark_result.throughput_gbps;
            ss << " | " << r.speedup_vs_reference << "x |\n";
        }

        return ss.str();
    }

    /**
     * @brief Save report to file
     */
    bool save_to_file(const std::string& path, const std::string& format = "json") const {
        std::ofstream file(path);
        if (!file.is_open()) {
            return false;
        }

        if (format == "json") {
            file << to_json();
        } else if (format == "csv") {
            file << to_csv();
        } else if (format == "md" || format == "markdown") {
            file << to_markdown();
        } else {
            return false;
        }

        return true;
    }

    /**
     * @brief Print summary to console
     */
    void print_summary() const {
        int passed = 0, failed = 0;
        for (const auto& r : results_) {
            if (r.correctness_passed) ++passed;
            else ++failed;
        }

        std::cout << "\n";
        std::cout << "============================================\n";
        std::cout << "  Test Report Summary - " << backend_name_ << "\n";
        std::cout << "============================================\n";
        std::cout << "  Timestamp: " << timestamp_ << "\n";
        std::cout << "  Total:     " << results_.size() << "\n";
        std::cout << "  Passed:    " << passed << "\n";
        std::cout << "  Failed:    " << failed << "\n";
        std::cout << "  Pass Rate: " << std::fixed << std::setprecision(1)
                  << (100.0 * passed / results_.size()) << "%\n";
        std::cout << "============================================\n\n";

        // Print failed tests
        if (failed > 0) {
            std::cout << "Failed Tests:\n";
            for (const auto& r : results_) {
                if (!r.correctness_passed) {
                    std::cout << "  - " << r.operator_name << ": "
                              << r.error_message << "\n";
                }
            }
            std::cout << "\n";
        }
    }

    /**
     * @brief Get test report structure
     */
    TestReport get_report() const {
        TestReport report;
        report.backend = backend_name_;
        report.timestamp = timestamp_;
        report.results = results_;
        report.total_tests = static_cast<int>(results_.size());
        report.passed_tests = 0;
        report.failed_tests = 0;

        for (const auto& r : results_) {
            if (r.correctness_passed) {
                ++report.passed_tests;
            } else {
                ++report.failed_tests;
            }
            report.category_total_counts[r.category]++;
            if (r.correctness_passed) {
                report.category_pass_counts[r.category]++;
            }
        }

        return report;
    }

private:
    std::string backend_name_;
    std::string timestamp_;
    std::vector<OperatorTestResult> results_;
};

/**
 * @brief Create a standardized test result entry
 */
inline OperatorTestResult create_result(
    const std::string& op_name,
    const std::string& category,
    const test::CorrectnessResult& correctness,
    const test::BenchmarkResult& benchmark,
    double speedup = 1.0) {

    OperatorTestResult result;
    result.operator_name = op_name;
    result.category = category;
    result.correctness_passed = correctness.passed;
    result.correctness_detail = correctness;
    result.benchmark_result = benchmark;
    result.speedup_vs_reference = speedup;
    result.error_message = correctness.passed ? "" : correctness.message;
    return result;
}

} // namespace report
} // namespace triton_jit

#endif // TRITON_JIT_REPORT_GENERATOR_H
