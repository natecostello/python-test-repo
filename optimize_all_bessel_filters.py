#!/usr/bin/env python3
"""
Multi-Order Bessel Filter Optimization Script

This script optimizes the design frequency ratios for 2nd, 4th, 8th, and 12th order
Bessel filters to achieve -3dB frequencies within tight tolerance of 4000 Hz target.
"""

import numpy as np
from scipy import signal


def find_frequency_at_db(freq, mag_db, target_db, tolerance=0.001):
    """Find frequency where magnitude crosses target dB level with
    ultra-high precision"""
    diff = np.abs(mag_db - target_db)
    idx = np.argmin(diff)

    if diff[idx] <= tolerance:
        return freq[idx]
    else:
        # Ultra-high precision linear interpolation
        if idx > 0 and idx < len(mag_db) - 1:
            if mag_db[idx] > target_db:
                idx1, idx2 = idx, idx + 1
            else:
                idx1, idx2 = idx - 1, idx

            x1, x2 = freq[idx1], freq[idx2]
            y1, y2 = mag_db[idx1], mag_db[idx2]
            freq_interp = x1 + (target_db - y1) * (x2 - x1) / (y2 - y1)
            return freq_interp
    return None


def test_bessel_filter_ratio(ratio, target_3db_freq=4000, order=2, verbose=False):
    """Test a specific ratio and return the actual -3dB frequency"""
    # Calculate design frequency
    design_freq = target_3db_freq / ratio

    # Design Bessel filter
    try:
        b, a = signal.bessel(order, 2 * np.pi * design_freq, analog=True)
    except Exception as e:
        if verbose:
            print(
                f"Error designing {order}th order filter with ratio "
                f"{ratio:.6f}: {e}"
            )
        return None

    # Create ultra-high-resolution frequency range around target
    frequencies_fine = np.linspace(1, 10000, 30000)  # Ultra-high resolution
    omega_fine = 2 * np.pi * frequencies_fine

    # Calculate frequency response
    try:
        w, h = signal.freqs(b, a, worN=omega_fine)
    except Exception as e:
        if verbose:
            print(
                f"Error calculating frequency response for {order}th order "
                f"with ratio {ratio:.6f}: {e}"
            )
        return None

    # Convert to Hz and dB
    freq_hz = w / (2 * np.pi)
    mag_db = 20 * np.log10(np.abs(h))

    # Find -3dB frequency
    freq_3db = find_frequency_at_db(freq_hz, mag_db, -3.0)

    if verbose and freq_3db:
        error = freq_3db - target_3db_freq
        print(
            f"Order {order}: Ratio {ratio:.6f}, Design freq: "
            f"{design_freq:.1f} Hz, "
            f"Actual -3dB: {freq_3db:.3f} Hz, Error: {error:+.3f} Hz"
        )

    return freq_3db


def optimize_bessel_filter_ratio(
    order, target_3db_freq=4000, tolerance=0.5, max_iterations=100
):
    """
    Optimize the ratio for a specific order Bessel filter

    Parameters:
    - order: Filter order (2, 4, 8, 12)
    - target_3db_freq: Target -3dB frequency in Hz (default 4000)
    - tolerance: Acceptable error in Hz (default 0.5)
    - max_iterations: Maximum number of iterations (default 100)

    Returns:
    - optimized_ratio: The best ratio found
    - actual_freq: The actual -3dB frequency achieved
    - error: The error from target
    """

    print(f"\nOptimizing {order}th Order Bessel Filter")
    print(f"Target -3dB frequency: {target_3db_freq} Hz")
    print(f"Tolerance: ±{tolerance} Hz")
    print("=" * 50)

    # Set initial search bounds based on order
    if order == 2:
        ratio_low = 0.78  # Lower bound for 2nd order
        ratio_high = 0.79  # Upper bound for 2nd order
    elif order == 4:
        ratio_low = 0.658  # Lower bound for 4th order
        ratio_high = 0.662  # Upper bound for 4th order
    elif order == 8:
        ratio_low = 0.514  # Lower bound for 8th order
        ratio_high = 0.518  # Upper bound for 8th order
    elif order == 12:
        ratio_low = 0.434  # Lower bound for 12th order
        ratio_high = 0.436  # Upper bound for 12th order
    else:
        print(f"Unsupported order: {order}")
        return None, None, None

    best_ratio = None
    best_freq = None
    best_error = float("inf")

    for iteration in range(max_iterations):
        # Binary search approach
        ratio_mid = (ratio_low + ratio_high) / 2

        # Test the middle point
        freq_mid = test_bessel_filter_ratio(ratio_mid, target_3db_freq, order)

        if freq_mid is None:
            print(
                f"Iteration {iteration + 1}: Failed to calculate for ratio "
                f"{ratio_mid:.6f}"
            )
            break

        error_mid = freq_mid - target_3db_freq
        abs_error_mid = abs(error_mid)

        # Update best result if this is better
        if abs_error_mid < abs(best_error):
            best_ratio = ratio_mid
            best_freq = freq_mid
            best_error = error_mid

        print(
            f"Iter {iteration + 1:2d}: Ratio={ratio_mid:.7f}, "
            f"Actual -3dB={freq_mid:8.3f} Hz, Error={error_mid:+8.3f} Hz"
        )

        # Check if we've met the tolerance
        if abs_error_mid <= tolerance:
            print(f"✅ SUCCESS! Tolerance met in {iteration + 1} iterations")
            return ratio_mid, freq_mid, error_mid

        # Narrow the search range based on error direction
        if error_mid > 0:  # Frequency too high, need higher ratio
            ratio_low = ratio_mid
        else:  # Frequency too low, need lower ratio
            ratio_high = ratio_mid

        # Check if search range is getting too narrow
        if (ratio_high - ratio_low) < 1e-9:
            print("Search range too narrow. Stopping optimization.")
            break

    # Return the best result found
    print(f"Best result: Ratio {best_ratio:.7f}, Error: {best_error:+.3f} Hz")
    return best_ratio, best_freq, best_error


def optimize_all_orders(target_3db_freq=4000, tolerance=0.5):
    """Optimize all filter orders and return the results"""
    orders = [2, 4, 8, 12]
    results = {}

    print("Multi-Order Bessel Filter Optimization")
    print(f"Target: {target_3db_freq} Hz, Tolerance: ±{tolerance} Hz")
    print("=" * 70)

    for order in orders:
        ratio, freq, error = optimize_bessel_filter_ratio(
            order, target_3db_freq, tolerance, max_iterations=100
        )

        if ratio is not None:
            results[order] = {
                "ratio": ratio,
                "freq_3db": freq,
                "error": error,
                "tolerance_met": (
                    abs(error) <= tolerance if error is not None else False
                ),
            }
        else:
            results[order] = {
                "ratio": None,
                "freq_3db": None,
                "error": None,
                "tolerance_met": False,
            }

    return results


def display_optimization_results(results, target_3db_freq=4000, tolerance=0.5):
    """Display comprehensive optimization results"""
    print(f"\n{'=' * 80}")
    print("COMPREHENSIVE OPTIMIZATION RESULTS")
    print("=" * 80)

    print(f"Target -3dB frequency: {target_3db_freq} Hz")
    print(f"Tolerance requirement: ±{tolerance} Hz")
    print(
        f"\n{'Order':<6} {'Ratio':<12} {'Design Freq':<12} "
        f"{'Actual -3dB':<12} {'Error':<10} {'Status'}"
    )
    print("-" * 70)

    for order in sorted(results.keys()):
        result = results[order]
        if result["ratio"] is not None:
            design_freq = target_3db_freq / result["ratio"]
            status = "✅ PASS" if result["tolerance_met"] else "❌ FAIL"
            print(
                f"{order:<6} {result['ratio']:<12.7f} "
                f"{design_freq:<12.1f} "
                f"{result['freq_3db']:<12.3f} {result['error']:<+10.3f} "
                f"{status}"
            )
        else:
            print(
                f"{order:<6} {'Failed':<12} {'Failed':<12} "
                f"{'Failed':<12} {'Failed':<10} {'❌ FAIL'}"
            )

    # Generate optimized ratios dictionary for notebook update
    print(f"\n{'=' * 80}")
    print("OPTIMIZED RATIOS FOR NOTEBOOK UPDATE:")
    print("=" * 80)
    print("ratios = {")

    for order in sorted(results.keys()):
        result = results[order]
        if result["ratio"] is not None:
            error_str = (
                f"±{abs(result['error']):.2f} Hz"
                if result["error"] is not None
                else "Failed"
            )
            suffix = "nd" if order == 2 else "th"
            print(
                f"    {order}: {result['ratio']:.7f},  # Optimized ratio for "
                f"{order}{suffix} order ({error_str})"
            )
        else:
            suffix = "nd" if order == 2 else "th"
            print(
                f"    {order}: 0.5000000,  # Failed optimization for "
                f"{order}{suffix} order"
            )

    print("}")

    # Summary statistics
    successful_orders = [
        order for order, result in results.items() if result["tolerance_met"]
    ]

    print("\n📊 SUMMARY:")
    print(f"Orders meeting tolerance: {len(successful_orders)}/{len(results)}")
    if successful_orders:
        print(f"Successful orders: {', '.join(map(str, successful_orders))}")

        best_accuracy = min(
            abs(results[order]["error"])
            for order in successful_orders
            if results[order]["error"] is not None
        )
        best_order = min(
            successful_orders,
            key=lambda x: (
                abs(results[x]["error"])
                if results[x]["error"] is not None
                else float("inf")
            ),
        )

        print(f"Best accuracy: ±{best_accuracy:.3f} Hz (Order {best_order})")

    failed_orders = [
        order for order, result in results.items() if not result["tolerance_met"]
    ]
    if failed_orders:
        print(f"Orders needing further work: {', '.join(map(str, failed_orders))}")


if __name__ == "__main__":
    print("Multi-Order Bessel Filter Optimization")
    print("======================================\n")

    # Run optimization for all orders
    results = optimize_all_orders(target_3db_freq=4000, tolerance=0.5)

    # Display comprehensive results
    display_optimization_results(results, target_3db_freq=4000, tolerance=0.5)

    # Check overall success
    all_passed = all(result["tolerance_met"] for result in results.values())

    if all_passed:
        print("\n🎉 ALL FILTERS OPTIMIZED SUCCESSFULLY!")
        print("All orders now meet the ±0.5 Hz tolerance requirement.")
    else:
        passed_count = sum(1 for result in results.values() if result["tolerance_met"])
        print(
            f"\n⚠️  PARTIAL SUCCESS: {passed_count}/{len(results)} "
            f"filters optimized successfully."
        )
        print(
            "Consider relaxing tolerance or using different optimization "
            "approach for failed filters."
        )
