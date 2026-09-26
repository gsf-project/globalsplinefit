"""Performance tests for the new GSF API."""

import time

import numpy as np
import pytest


class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks for GSF models."""

    def test_single_energy_point_performance(self, gsf_energy):
        """Benchmark single energy point calculation speed."""
        energy = np.array([100.0])

        start_time = time.perf_counter()
        for _ in range(1000):  # 1000 iterations
            _ = gsf_energy.flux(energy, "p")
        end_time = time.perf_counter()

        avg_time_per_call = (end_time - start_time) / 1000
        # Should be very fast for single point
        assert avg_time_per_call < 0.01, (
            f"Single point calculation too slow: {avg_time_per_call * 1000:.2f}ms per call"
        )

    def test_medium_array_performance(self, gsf_energy):
        """Benchmark medium-sized array calculations."""
        energies = np.logspace(0, 5, 1000)  # 1000 points

        # Run multiple times for measurable results
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            flux = gsf_energy.flux(energies, "p")
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        min_time = min(times)
        avg_time = sum(times) / len(times)

        assert min_time < 1.0, (
            f"Medium array calculation too slow: min={min_time:.3f}s, avg={avg_time:.3f}s"
        )
        assert len(flux) == len(energies)

    def test_large_array_performance(self, gsf_energy):
        """Benchmark large array calculations."""
        energies = np.logspace(0, 6, 10000)  # 10,000 points

        # Run multiple times for measurable results
        times = []
        for _ in range(5):  # Fewer iterations for large arrays
            start_time = time.perf_counter()
            flux = gsf_energy.flux(energies, "p")
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        min_time = min(times)
        avg_time = sum(times) / len(times)

        assert min_time < 5.0, (
            f"Large array calculation too slow: min={min_time:.3f}s, avg={avg_time:.3f}s"
        )
        assert len(flux) == len(energies)

    def test_all_groups_performance(self, gsf_energy):
        """Benchmark calculating all groups."""
        energies = np.logspace(0, 4, 500)  # Moderate size
        groups = gsf_energy.active_groups

        # Run multiple times for measurable results
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            fluxes = {}
            for group in groups:
                fluxes[group] = gsf_energy.flux(energies, group)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        min_time = min(times)
        avg_time = sum(times) / len(times)
        avg_time_per_group = avg_time / len(groups)

        assert min_time < 3.0, (
            f"All groups calculation too slow: min={min_time:.3f}s, avg={avg_time:.3f}s"
        )
        assert avg_time_per_group < 1.0, (
            f"Average per group too slow: {avg_time_per_group:.3f}s"
        )

    def test_total_flux_vs_sum_performance(self, gsf_energy):
        """Compare performance of total_flux method vs manual summing."""
        energies = np.logspace(1, 4, 1000)

        # Method 1: Using total_flux
        start_time = time.perf_counter()
        total_flux_method = gsf_energy.total_flux(energies)
        method1_time = time.perf_counter() - start_time

        # Method 2: Manual sum
        start_time = time.perf_counter()
        manual_sum = (
            gsf_energy.flux(energies, "H")
            + gsf_energy.flux(energies, "He")
            + gsf_energy.flux(energies, "O*")
            + gsf_energy.flux(energies, "Fe*")
        )
        method2_time = time.perf_counter() - start_time

        # total_flux should be competitive or faster
        speedup_ratio = method2_time / method1_time
        assert speedup_ratio > 0.5, (
            f"total_flux method much slower than manual sum: {speedup_ratio:.2f}x"
        )

        # Results should be reasonably similar (total_flux sums individual elements,
        # manual_sum sums groups which include multiple elements)
        assert np.all(total_flux_method <= manual_sum), (
            "Total flux should be <= sum of group fluxes"
        )
        relative_diff = np.abs(total_flux_method - manual_sum) / manual_sum
        assert np.all(relative_diff < 0.5), "Relative difference should be < 50%"

    def test_error_calculation_performance(self, gsf_energy):
        """Benchmark error calculation performance."""
        energies = np.logspace(1, 4, 100)  # Smaller for error calc

        # Single group error - run multiple times
        single_times = []
        for _ in range(3):
            start_time = time.perf_counter()
            _ = gsf_energy.error(energies, "p")
            single_times.append(time.perf_counter() - start_time)

        # Total error - run multiple times
        total_times = []
        for _ in range(3):  # Fewer iterations as this is slower
            start_time = time.perf_counter()
            _ = gsf_energy.total_error(energies)
            total_times.append(time.perf_counter() - start_time)

        single_min = min(single_times)
        single_avg = sum(single_times) / len(single_times)
        total_min = min(total_times)
        total_avg = sum(total_times) / len(total_times)

        assert single_min < 2.0, (
            f"Single group error calculation too slow: min={single_min:.3f}s, avg={single_avg:.3f}s"
        )
        assert total_min < 5.0, (
            f"Total error calculation too slow: min={total_min:.3f}s, avg={total_avg:.3f}s"
        )

    def test_covariance_calculation_performance(self, gsf_energy):
        """Benchmark covariance calculation performance."""
        energies = np.logspace(2, 4, 100)  # Small for covariance

        # Run multiple times for measurable results
        times = []
        for _ in range(3):  # Fewer iterations as covariance is expensive
            start_time = time.perf_counter()
            cov = gsf_energy.covariance("p", "p", energies)
            times.append(time.perf_counter() - start_time)

        min_time = min(times)
        avg_time = sum(times) / len(times)

        assert min_time < 3.0, (
            f"Covariance calculation too slow: min={min_time:.3f}s, avg={avg_time:.3f}s"
        )
        assert cov.shape == (len(energies), len(energies))

    def test_jacobian_calculation_performance(self, gsf_energy):
        """Benchmark Jacobian calculation performance."""
        energies = np.logspace(2, 4, 200)

        # Run multiple times for measurable results
        times = []
        for _ in range(5):
            start_time = time.perf_counter()
            jac = gsf_energy.jacobian(energies, "p")
            times.append(time.perf_counter() - start_time)

        min_time = min(times)
        avg_time = sum(times) / len(times)

        assert min_time < 2.0, (
            f"Jacobian calculation too slow: min={min_time:.3f}s, avg={avg_time:.3f}s"
        )
        assert jac.shape[0] == len(energies)

    def test_solar_modulation_performance(self, gsf_energy):
        """Benchmark solar modulation calculation performance."""
        energies = np.logspace(0, 4, 1000)
        time_interval = (200901, 200912)

        # LIS calculation - run multiple times
        lis_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            _ = gsf_energy.flux(energies, "p")
            lis_times.append(time.perf_counter() - start_time)

        # Solar modulation calculation - run multiple times
        solar_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            _ = gsf_energy.flux(energies, "p", time_interval=time_interval)
            solar_times.append(time.perf_counter() - start_time)

        lis_min = min(lis_times)
        solar_min = min(solar_times)

        # Solar modulation should not be dramatically slower
        slowdown_factor = solar_min / lis_min
        assert slowdown_factor < 15.0, (
            f"Solar modulation too much slower: {slowdown_factor:.1f}x"
        )

    def test_rigidity_model_performance(self, gsf_rigidity):
        """Benchmark rigidity model performance."""
        rigidities = np.logspace(-1, 3, 1000)

        # Run multiple times for measurable results
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            _ = gsf_rigidity.flux(rigidities, "p")
            times.append(time.perf_counter() - start_time)

        min_time = min(times)
        avg_time = sum(times) / len(times)

        assert min_time < 2.0, (
            f"Rigidity flux calculation too slow: min={min_time:.3f}s, avg={avg_time:.3f}s"
        )

    def test_nucleon_model_performance(self, gsf_nucleon):
        """Benchmark nucleon model performance."""
        energies_per_nucleon = np.logspace(0, 3, 200)

        # Run multiple times for measurable results
        times = []
        for _ in range(3):
            start_time = time.perf_counter()
            flux = gsf_nucleon.flux(energies_per_nucleon, "p")
            times.append(time.perf_counter() - start_time)

        min_time = min(times)
        avg_time = sum(times) / len(times)

        assert min_time < 2.0, (
            f"Nucleon flux calculation too slow: min={min_time:.3f}s, avg={avg_time:.3f}s"
        )
        assert flux.shape == (len(energies_per_nucleon),)

    def test_repeated_calculation_consistency(self, gsf_energy):
        """Test that repeated calculations are consistent and don't slow down."""
        energies = np.logspace(1, 4, 500)

        times = []
        results = []

        for _ in range(5):
            start_time = time.perf_counter()
            flux = gsf_energy.flux(energies, "p")
            end_time = time.perf_counter()

            times.append(end_time - start_time)
            results.append(flux.copy())

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0], results[i], err_msg=f"Run {i} differs from run 0"
            )

        # Times should be consistent (no significant slowdown)
        avg_time = np.mean(times)
        max_time = np.max(times)
        assert max_time < 3 * avg_time, (
            f"Inconsistent timing: max={max_time:.3f}s, avg={avg_time:.3f}s"
        )

    def test_memory_usage_stability(self, gsf_energy):
        """Test that memory usage remains stable across multiple calculations."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            # Perform many calculations
            energies = np.logspace(1, 4, 1000)
            for i in range(20):
                flux = gsf_energy.flux(energies, "p")
                del flux  # Explicitly delete

                if i % 5 == 0:  # Check memory every 5 iterations
                    current_memory = process.memory_info().rss
                    memory_increase = current_memory - initial_memory
                    # Memory should not grow significantly (< 50MB increase)
                    assert memory_increase < 50 * 1024 * 1024, (
                        f"Memory leak detected: {memory_increase / 1024 / 1024:.1f}MB increase at iteration {i}"
                    )

        except ImportError:
            # Skip if psutil not available
            pytest.skip("psutil not available for memory testing")

    def test_parallel_calculation_simulation(self, gsf_energy):
        """Simulate parallel-like calculations to test for race conditions."""
        energies = np.logspace(1, 4, 200)

        # Simulate what might happen with threading by doing rapid alternating calculations
        start_time = time.perf_counter()

        results = []
        for i in range(10):
            # Alternate between different groups rapidly
            group = gsf_energy.active_groups[i % 4]
            flux = gsf_energy.flux(energies, group)
            results.append((group, flux.copy()))

        end_time = time.perf_counter()

        total_time = end_time - start_time
        assert total_time < 5.0, (
            f"Simulated parallel calculations too slow: {total_time:.2f}s"
        )

        # Verify results are consistent for same groups
        p_results = [flux for group, flux in results if group == "p"]
        he_results = [flux for group, flux in results if group == "He"]

        if len(p_results) > 1:
            for i in range(1, len(p_results)):
                np.testing.assert_array_equal(
                    p_results[0], p_results[i], err_msg="Proton results not consistent"
                )

        if len(he_results) > 1:
            for i in range(1, len(he_results)):
                np.testing.assert_array_equal(
                    he_results[0],
                    he_results[i],
                    err_msg="Helium results not consistent",
                )

    def test_scaling_behavior(self, gsf_energy):
        """Test how calculation time scales with array size."""
        sizes = [100, 500, 1000, 2000]
        times = []

        for size in sizes:
            energies = np.logspace(1, 4, size)

            # Best-of-N timing: at these array sizes a single flux() call is
            # sub-millisecond, so fixed Python/numpy overhead dominates and the
            # per-size measurement is noisy on shared CI runners. The minimum
            # over repeats is the most robust estimator of the true cost.
            size_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                flux = gsf_energy.flux(energies, "p")
                size_times.append(time.perf_counter() - start_time)

            times.append(min(size_times))
            assert len(flux) == size

        # Spline evaluation is O(n); scaling should be roughly linear. The
        # adjacent-pair ratio still carries substantial jitter at these tiny
        # absolute times, so the guard is generous (a genuine super-quadratic
        # regression would blow the ratio far past this bound), while tolerating
        # the scheduling noise of shared runners (observed ~4.4x on a 2x step).
        for i in range(1, len(times)):
            size_ratio = sizes[i] / sizes[i - 1]
            time_ratio = times[i] / times[i - 1]

            assert time_ratio < size_ratio * 3.0, (
                f"Poor scaling: {size_ratio:.1f}x size increase caused {time_ratio:.1f}x time increase"
            )
