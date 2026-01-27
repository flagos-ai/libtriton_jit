#!/usr/bin/env python3
"""
Triton JIT Operator Test Runner

Runs all operator tests and generates reports.

Usage:
    python run_all_tests.py --backend CUDA --output reports/cuda_results.json
    python run_all_tests.py --backend MUSA --config configs/quick.json
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import time


class TestRunner:
    """Runs operator tests and collects results."""

    OPERATOR_CATEGORIES = [
        "pointwise",
        "reduce",
        "matmul",
        "index",
        "normalization",
        "fusion",
    ]

    OPERATORS = {
        "pointwise": ["add", "fill", "zeros", "exponential_", "contiguous"],
        "reduce": ["sum", "max", "argmax", "topk"],
        "matmul": ["mm", "bmm", "addmm"],
        "index": ["embedding", "nonzero", "cat", "reshape_and_cache_flash"],
        "normalization": ["softmax", "rms_norm", "fused_add_rms_norm"],
        "fusion": ["apply_rotary_pos_emb", "rwkv_ka_fusion", "rwkv_mm_sparsity"],
    }

    def __init__(self, build_dir: str, backend: str, timeout: int = 300):
        self.build_dir = Path(build_dir).resolve()
        self.backend = backend
        self.timeout = timeout
        self.results: List[Dict[str, Any]] = []

    def find_test_executable(self, category: str, op_name: str) -> Path:
        """Find the test executable for an operator."""
        # Try different possible paths
        paths = [
            self.build_dir / "operators" / category / op_name / f"test_{op_name}",
            self.build_dir / "operators" / category / f"test_{op_name}",
        ]

        for path in paths:
            if path.exists():
                return path

        return paths[0]  # Return default path even if not found

    def run_single_test(self, category: str, op_name: str) -> Dict[str, Any]:
        """Run a single test and return results."""
        result = {
            "operator": op_name,
            "category": category,
            "passed": False,
            "output": "",
            "error": "",
            "duration_s": 0,
        }

        test_path = self.find_test_executable(category, op_name)

        if not test_path.exists():
            result["error"] = f"Test executable not found: {test_path}"
            return result

        try:
            start_time = time.time()
            proc = subprocess.run(
                [str(test_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=test_path.parent,
            )
            end_time = time.time()

            result["duration_s"] = end_time - start_time
            result["output"] = proc.stdout
            result["error"] = proc.stderr
            result["passed"] = proc.returncode == 0

        except subprocess.TimeoutExpired:
            result["error"] = f"Test timed out after {self.timeout}s"
        except Exception as e:
            result["error"] = str(e)

        return result

    def run_all_tests(self, categories: List[str] = None, operators: List[str] = None) -> List[Dict[str, Any]]:
        """Run all tests or a subset."""
        if categories is None:
            categories = self.OPERATOR_CATEGORIES

        self.results = []

        for category in categories:
            if category not in self.OPERATORS:
                print(f"Warning: Unknown category {category}")
                continue

            ops = self.OPERATORS[category]
            if operators:
                ops = [op for op in ops if op in operators]

            for op_name in ops:
                print(f"Running test: {category}/{op_name}... ", end="", flush=True)
                result = self.run_single_test(category, op_name)
                self.results.append(result)

                if result["passed"]:
                    print(f"PASSED ({result['duration_s']:.2f}s)")
                else:
                    print(f"FAILED")
                    if result["error"]:
                        print(f"  Error: {result['error'][:200]}")

        return self.results

    def generate_report(self) -> Dict[str, Any]:
        """Generate a summary report."""
        passed = sum(1 for r in self.results if r["passed"])
        failed = sum(1 for r in self.results if not r["passed"])

        report = {
            "backend": self.backend,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(self.results),
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / len(self.results) * 100 if self.results else 0,
            },
            "results": self.results,
        }

        # Category breakdown
        category_stats = {}
        for r in self.results:
            cat = r["category"]
            if cat not in category_stats:
                category_stats[cat] = {"passed": 0, "failed": 0}
            if r["passed"]:
                category_stats[cat]["passed"] += 1
            else:
                category_stats[cat]["failed"] += 1

        report["category_stats"] = category_stats

        return report

    def print_summary(self):
        """Print a summary to console."""
        report = self.generate_report()

        print("\n" + "=" * 60)
        print(f"  Test Summary - {self.backend} Backend")
        print("=" * 60)
        print(f"  Timestamp: {report['timestamp']}")
        print(f"  Total:     {report['summary']['total']}")
        print(f"  Passed:    {report['summary']['passed']}")
        print(f"  Failed:    {report['summary']['failed']}")
        print(f"  Pass Rate: {report['summary']['pass_rate']:.1f}%")
        print("=" * 60)

        print("\nCategory Breakdown:")
        for cat, stats in report["category_stats"].items():
            total = stats["passed"] + stats["failed"]
            rate = stats["passed"] / total * 100 if total > 0 else 0
            print(f"  {cat:20s}: {stats['passed']}/{total} ({rate:.0f}%)")

        if report["summary"]["failed"] > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  - {r['category']}/{r['operator']}")


def main():
    parser = argparse.ArgumentParser(description="Run Triton JIT operator tests")
    parser.add_argument(
        "--backend",
        choices=["CUDA", "MUSA", "NPU", "IX"],
        default="CUDA",
        help="Backend to test",
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        help="Build directory path",
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--config",
        help="Test configuration JSON file",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Run only specified categories",
    )
    parser.add_argument(
        "--operator",
        action="append",
        dest="operators",
        help="Run only specified operators",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Test timeout in seconds",
    )

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
            if "categories" in config and not args.categories:
                args.categories = config["categories"]
            if "operators" in config and not args.operators:
                args.operators = config["operators"]
            if "timeout" in config:
                args.timeout = config["timeout"]

    # Run tests
    runner = TestRunner(args.build_dir, args.backend, args.timeout)
    runner.run_all_tests(args.categories, args.operators)
    runner.print_summary()

    # Save report
    if args.output:
        report = runner.generate_report()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")

    # Return exit code based on test results
    return 0 if all(r["passed"] for r in runner.results) else 1


if __name__ == "__main__":
    sys.exit(main())
