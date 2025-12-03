"""
Timing and Benchmarking Utilities

This module provides utilities for measuring code execution time
and computing performance metrics like speedup ratios.

Features:
- Timer context manager for easy timing
- Function decorator for automatic timing
- Speedup calculation utilities
- Benchmark result containers

Example:
    >>> with Timer("MyOperation") as t:
    ...     result = expensive_computation()
    >>> print(f"Took {t.elapsed_ms:.2f} ms")
    
    >>> @time_function
    ... def my_function():
    ...     return do_work()
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, List, Dict
from functools import wraps
import statistics


class Timer:
    """
    Context manager for timing code blocks.
    
    Provides high-resolution timing using time.perf_counter().
    
    Attributes:
        name: Optional name for the timed operation
        elapsed: Elapsed time in seconds
        elapsed_ms: Elapsed time in milliseconds
        elapsed_us: Elapsed time in microseconds
    
    Example:
        >>> with Timer("Matrix multiply") as t:
        ...     result = np.dot(A, B)
        >>> print(f"Elapsed: {t.elapsed_ms:.2f} ms")
        Matrix multiply: 15.23 ms
    """
    
    def __init__(self, name: Optional[str] = None, verbose: bool = True):
        """
        Initialize timer.
        
        Args:
            name: Optional name to print with timing
            verbose: Whether to print timing on exit
        """
        self.name = name
        self.verbose = verbose
        self._start: float = 0
        self._end: float = 0
        self.elapsed: float = 0
    
    def __enter__(self) -> 'Timer':
        """Start the timer."""
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        """Stop the timer and optionally print result."""
        self._end = time.perf_counter()
        self.elapsed = self._end - self._start
        
        if self.verbose and self.name:
            print(f"{self.name}: {self.elapsed_ms:.2f} ms")
    
    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000
    
    @property
    def elapsed_us(self) -> float:
        """Elapsed time in microseconds."""
        return self.elapsed * 1_000_000
    
    def reset(self) -> None:
        """Reset the timer."""
        self._start = 0
        self._end = 0
        self.elapsed = 0
    
    def start(self) -> None:
        """Manually start the timer."""
        self._start = time.perf_counter()
    
    def stop(self) -> float:
        """Manually stop the timer and return elapsed time."""
        self._end = time.perf_counter()
        self.elapsed = self._end - self._start
        return self.elapsed


def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Prints execution time after each call.
    
    Example:
        >>> @time_function
        ... def slow_function(n):
        ...     time.sleep(n)
        ...     return n
        >>> slow_function(0.1)
        slow_function: 100.23 ms
        0.1
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        with Timer(func.__name__):
            result = func(*args, **kwargs)
        return result
    return wrapper


def time_function_silent(func: Callable) -> Callable:
    """
    Decorator to time function execution without printing.
    
    The timing is stored in func.last_elapsed_ms attribute.
    
    Example:
        >>> @time_function_silent
        ... def my_func():
        ...     return compute()
        >>> result = my_func()
        >>> print(f"Took {my_func.last_elapsed_ms:.2f} ms")
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        wrapper.last_elapsed = elapsed
        wrapper.last_elapsed_ms = elapsed * 1000
        return result
    
    wrapper.last_elapsed = 0.0
    wrapper.last_elapsed_ms = 0.0
    return wrapper


def compute_speedup(baseline_time: float, optimized_time: float) -> float:
    """
    Compute speedup ratio between baseline and optimized times.
    
    Speedup = baseline_time / optimized_time
    
    A speedup > 1 means the optimized version is faster.
    A speedup < 1 means the optimized version is slower.
    
    Args:
        baseline_time: Time for baseline implementation
        optimized_time: Time for optimized implementation
    
    Returns:
        Speedup ratio
    
    Example:
        >>> speedup = compute_speedup(100.0, 25.0)
        >>> print(f"Speedup: {speedup:.2f}x")
        Speedup: 4.00x
    """
    if optimized_time <= 0:
        return float('inf')
    return baseline_time / optimized_time


def compute_efficiency(speedup: float, num_processors: int) -> float:
    """
    Compute parallel efficiency.
    
    Efficiency = Speedup / NumProcessors
    
    Efficiency of 1.0 (100%) means perfect linear speedup.
    
    Args:
        speedup: Observed speedup
        num_processors: Number of parallel processors used
    
    Returns:
        Efficiency ratio (0-1, can exceed 1 for superlinear speedup)
    """
    if num_processors <= 0:
        return 0.0
    return speedup / num_processors


@dataclass
class BenchmarkResult:
    """
    Container for benchmark results.
    
    Stores multiple timing trials and computes statistics.
    
    Attributes:
        name: Name of the benchmarked operation
        times_ms: List of timing results in milliseconds
        metadata: Optional additional information
    """
    name: str
    times_ms: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_trial(self, time_ms: float) -> None:
        """Add a timing trial result."""
        self.times_ms.append(time_ms)
    
    @property
    def mean_ms(self) -> float:
        """Mean time in milliseconds."""
        if not self.times_ms:
            return 0.0
        return statistics.mean(self.times_ms)
    
    @property
    def std_ms(self) -> float:
        """Standard deviation in milliseconds."""
        if len(self.times_ms) < 2:
            return 0.0
        return statistics.stdev(self.times_ms)
    
    @property
    def min_ms(self) -> float:
        """Minimum time in milliseconds."""
        if not self.times_ms:
            return 0.0
        return min(self.times_ms)
    
    @property
    def max_ms(self) -> float:
        """Maximum time in milliseconds."""
        if not self.times_ms:
            return 0.0
        return max(self.times_ms)
    
    @property
    def median_ms(self) -> float:
        """Median time in milliseconds."""
        if not self.times_ms:
            return 0.0
        return statistics.median(self.times_ms)
    
    @property
    def num_trials(self) -> int:
        """Number of timing trials."""
        return len(self.times_ms)
    
    def summary(self) -> str:
        """Generate summary string."""
        return (f"{self.name}: {self.mean_ms:.2f} ± {self.std_ms:.2f} ms "
                f"(n={self.num_trials}, min={self.min_ms:.2f}, max={self.max_ms:.2f})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'median_ms': self.median_ms,
            'num_trials': self.num_trials,
            'metadata': self.metadata
        }


class Benchmark:
    """
    Benchmark runner for comparing multiple implementations.
    
    Example:
        >>> bench = Benchmark("Nearest Neighbor Search")
        >>> bench.add_implementation("brute_force", brute_force_nn)
        >>> bench.add_implementation("kdtree", kdtree_nn)
        >>> results = bench.run(data, n_trials=5)
        >>> bench.print_comparison()
    """
    
    def __init__(self, name: str):
        """
        Initialize benchmark.
        
        Args:
            name: Name of the benchmark
        """
        self.name = name
        self.implementations: Dict[str, Callable] = {}
        self.results: Dict[str, BenchmarkResult] = {}
    
    def add_implementation(self, name: str, func: Callable) -> None:
        """
        Add an implementation to benchmark.
        
        Args:
            name: Name of the implementation
            func: Callable to benchmark
        """
        self.implementations[name] = func
        self.results[name] = BenchmarkResult(name)
    
    def run(
        self,
        *args,
        n_trials: int = 5,
        warmup: int = 1,
        **kwargs
    ) -> Dict[str, BenchmarkResult]:
        """
        Run the benchmark on all implementations.
        
        Args:
            *args: Arguments to pass to implementations
            n_trials: Number of timing trials
            warmup: Number of warmup runs (not timed)
            **kwargs: Keyword arguments to pass to implementations
        
        Returns:
            Dictionary mapping implementation names to results
        """
        for impl_name, func in self.implementations.items():
            # Warmup runs
            for _ in range(warmup):
                _ = func(*args, **kwargs)
            
            # Timed runs
            result = self.results[impl_name]
            for _ in range(n_trials):
                with Timer(verbose=False) as t:
                    _ = func(*args, **kwargs)
                result.add_trial(t.elapsed_ms)
        
        return self.results
    
    def print_comparison(self, baseline: Optional[str] = None) -> None:
        """
        Print comparison table of results.
        
        Args:
            baseline: Name of baseline implementation for speedup calculation
        """
        print(f"\nBenchmark: {self.name}")
        print("=" * 60)
        
        # Get baseline time
        if baseline is None:
            baseline = list(self.results.keys())[0]
        baseline_time = self.results[baseline].mean_ms
        
        print(f"{'Implementation':<20} {'Mean (ms)':>12} {'Std (ms)':>10} {'Speedup':>10}")
        print("-" * 60)
        
        for name, result in self.results.items():
            speedup = compute_speedup(baseline_time, result.mean_ms)
            speedup_str = f"{speedup:.2f}x" if name != baseline else "(baseline)"
            print(f"{name:<20} {result.mean_ms:>12.2f} {result.std_ms:>10.2f} {speedup_str:>10}")
        
        print()
    
    def get_speedups(self, baseline: str) -> Dict[str, float]:
        """
        Get speedup ratios relative to baseline.
        
        Args:
            baseline: Name of baseline implementation
        
        Returns:
            Dictionary mapping implementation names to speedups
        """
        baseline_time = self.results[baseline].mean_ms
        return {
            name: compute_speedup(baseline_time, result.mean_ms)
            for name, result in self.results.items()
        }


def benchmark_function(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    n_trials: int = 5,
    warmup: int = 1,
    name: Optional[str] = None
) -> BenchmarkResult:
    """
    Benchmark a single function.
    
    Args:
        func: Function to benchmark
        args: Positional arguments for function
        kwargs: Keyword arguments for function
        n_trials: Number of timing trials
        warmup: Number of warmup runs
        name: Optional name for result
    
    Returns:
        BenchmarkResult with timing statistics
    """
    if kwargs is None:
        kwargs = {}
    if name is None:
        name = func.__name__
    
    result = BenchmarkResult(name)
    
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    
    # Timed runs
    for _ in range(n_trials):
        with Timer(verbose=False) as t:
            _ = func(*args, **kwargs)
        result.add_trial(t.elapsed_ms)
    
    return result


if __name__ == "__main__":
    import numpy as np
    
    print("Timing Utilities Demo")
    print("=" * 50)
    
    # Demo Timer context manager
    print("\n1. Timer Context Manager:")
    with Timer("Sleep 100ms") as t:
        time.sleep(0.1)
    print(f"   Actual elapsed: {t.elapsed_ms:.2f} ms")
    
    # Demo function decorator
    print("\n2. Function Decorator:")
    
    @time_function
    def matrix_multiply(n):
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        return np.dot(A, B)
    
    _ = matrix_multiply(100)
    
    # Demo benchmark
    print("\n3. Benchmark Comparison:")
    
    def method_a(n):
        return sum(range(n))
    
    def method_b(n):
        return n * (n - 1) // 2
    
    bench = Benchmark("Sum of range")
    bench.add_implementation("loop", method_a)
    bench.add_implementation("formula", method_b)
    bench.run(1000000, n_trials=5)
    bench.print_comparison(baseline="loop")
    
    # Demo BenchmarkResult
    print("4. BenchmarkResult Statistics:")
    result = BenchmarkResult("test")
    for _ in range(10):
        result.add_trial(np.random.uniform(10, 20))
    print(f"   {result.summary()}")
