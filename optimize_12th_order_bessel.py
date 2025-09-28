#!/usr/bin/env python3
"""
12th Order Bessel Filter Optimization Script

This script iteratively optimizes the design frequency ratio for a 12th order
Bessel filter to achieve a -3dB frequency within 1.5 Hz of 4000 Hz target.
"""

import numpy as np
from scipy import signal


def find_frequency_at_db(freq, mag_db, target_db, tolerance=0.01):
    """Find frequency where magnitude crosses target dB level with high precision"""
    diff = np.abs(mag_db - target_db)
    idx = np.argmin(diff)

    if diff[idx] <= tolerance:
        return freq[idx]
    else:
        # Linear interpolation for better accuracy
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


def test_bessel_filter_ratio(ratio, target_3db_freq=4000, order=12, verbose=False):
    """Test a specific ratio and return the actual -3dB frequency"""
    # Calculate design frequency
    design_freq = target_3db_freq / ratio

    # Design Bessel filter
    try:
        b, a = signal.bessel(order, 2 * np.pi * design_freq, analog=True)
    except Exception as e:
        if verbose:
            print(f"Error designing filter with ratio {ratio:.6f}: {e}")
        return None

    # Create high-resolution frequency range around target
    frequencies_fine = np.linspace(1, 10000, 20000)  # Very high resolution
    omega_fine = 2 * np.pi * frequencies_fine

    # Calculate frequency response
    try:
        w, h = signal.freqs(b, a, worN=omega_fine)
    except Exception as e:
        if verbose:
            print(
                f"Error calculating frequency response with ratio " f"{ratio:.6f}: {e}"
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
            f"Ratio: {ratio:.6f}, Design freq: {design_freq:.1f} Hz, "
            f"Actual -3dB: {freq_3db:.2f} Hz, Error: {error:+.2f} Hz"
        )

    return freq_3db


def optimize_bessel_filter_ratio(
    target_3db_freq=4000, order=12, tolerance=1.5, max_iterations=50
):
    """
    Optimize the ratio for a Bessel filter to achieve target -3dB
    frequency within tolerance

    Parameters:
    - target_3db_freq: Target -3dB frequency in Hz (default 4000)
    - order: Filter order (default 12)
    - tolerance: Acceptable error in Hz (default 1.5)
    - max_iterations: Maximum number of iterations (default 50)

    Returns:
    - optimized_ratio: The best ratio found
    - actual_freq: The actual -3dB frequency achieved
    - error: The error from target
    """

    print(f"Optimizing {order}th Order Bessel Filter")
    print(f"Target -3dB frequency: {target_3db_freq} Hz")
    print(f"Tolerance: ±{tolerance} Hz")
    print(f"Maximum iterations: {max_iterations}")
    print("=" * 60)

    # Start with current estimate for 12th order
    ratio_low = 0.42  # Lower bound
    ratio_high = 0.44  # Upper bound

    best_ratio = None
    best_freq = None
    best_error = float("inf")

    for iteration in range(max_iterations):
        # Try three points: low, middle, high
        ratio_mid = (ratio_low + ratio_high) / 2

        # Test the middle point
        freq_mid = test_bessel_filter_ratio(ratio_mid, target_3db_freq, order)

        if freq_mid is None:
            print(
                f"Iteration {iteration + 1}: Failed to calculate frequency "
                f"response for ratio {ratio_mid:.6f}"
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
            f"Iteration {iteration + 1:2d}: Ratio={ratio_mid:.6f}, "
            f"Actual -3dB={freq_mid:7.2f} Hz, Error={error_mid:+7.2f} Hz"
        )

        # Check if we've met the tolerance
        if abs_error_mid <= tolerance:
            print(f"\n✅ SUCCESS! Tolerance met in {iteration + 1} iterations")
            print(f"Optimal ratio: {ratio_mid:.6f}")
            print(f"Actual -3dB frequency: {freq_mid:.2f} Hz")
            print(f"Error from target: {error_mid:+.2f} Hz")
            return ratio_mid, freq_mid, error_mid

        # Narrow the search range based on error direction
        if error_mid > 0:  # Frequency too high, need higher ratio
            ratio_low = ratio_mid
        else:  # Frequency too low, need lower ratio
            ratio_high = ratio_mid

        # Check if search range is getting too narrow
        if (ratio_high - ratio_low) < 1e-8:
            print("\nSearch range too narrow. Stopping optimization.")
            break

    # If we didn't meet tolerance, return the best result found
    print("\n⚠️  Maximum iterations reached without meeting tolerance")
    print(f"Best ratio found: {best_ratio:.6f}")
    print(f"Best -3dB frequency: {best_freq:.2f} Hz")
    print(f"Best error: {best_error:+.2f} Hz")

    return best_ratio, best_freq, best_error


def verify_optimization_result(ratio, target_3db_freq=4000, order=12):
    """Verify the optimization result with detailed analysis"""
    print("\n" + "=" * 60)
    print("VERIFICATION OF OPTIMIZED 12TH ORDER BESSEL FILTER")
    print("=" * 60)

    design_freq = target_3db_freq / ratio

    # Design filter
    b, a = signal.bessel(order, 2 * np.pi * design_freq, analog=True)

    # High-resolution frequency analysis
    frequencies_fine = np.linspace(1, 10000, 25000)
    omega_fine = 2 * np.pi * frequencies_fine

    w, h = signal.freqs(b, a, worN=omega_fine)
    freq_hz = w / (2 * np.pi)
    mag_db = 20 * np.log10(np.abs(h))

    # Find key frequencies
    freq_1db = find_frequency_at_db(freq_hz, mag_db, -1.0)
    freq_3db = find_frequency_at_db(freq_hz, mag_db, -3.0)
    freq_50db = find_frequency_at_db(freq_hz, mag_db, -50.0)

    error_3db = freq_3db - target_3db_freq if freq_3db else None

    print(f"Optimized ratio: {ratio:.6f}")
    print(f"Design frequency: {design_freq:.1f} Hz")
    print(f"Target -3dB frequency: {target_3db_freq} Hz")

    if freq_1db:
        print(f"-1 dB frequency: {freq_1db:.2f} Hz")
    if freq_3db:
        print(f"-3 dB frequency: {freq_3db:.2f} Hz")
        print(f"Error from target: {error_3db:+.2f} Hz")
        accuracy_percent = abs(error_3db) / target_3db_freq * 100
        print(f"Accuracy: {accuracy_percent:.4f}% error")
    if freq_50db:
        print(f"-50 dB frequency: {freq_50db:.1f} Hz")

    # Check tolerance
    tolerance_met = abs(error_3db) <= 1.5 if error_3db is not None else False
    print(f"\nTolerance (±1.5 Hz): {'✅ MET' if tolerance_met else '❌ NOT MET'}")

    return {
        "ratio": ratio,
        "design_freq": design_freq,
        "freq_1db": freq_1db,
        "freq_3db": freq_3db,
        "freq_50db": freq_50db,
        "error_3db": error_3db,
        "tolerance_met": tolerance_met,
    }


if __name__ == "__main__":
    print("12th Order Bessel Filter Optimization")
    print("=====================================\n")

    # Run optimization
    optimal_ratio, actual_freq, error = optimize_bessel_filter_ratio(
        target_3db_freq=4000, order=12, tolerance=1.5, max_iterations=50
    )

    # Verify result
    verification = verify_optimization_result(optimal_ratio, 4000, 12)

    # Output the final result for use in notebook
    print("\n" + "=" * 60)
    print("FINAL RESULT FOR NOTEBOOK UPDATE:")
    print("=" * 60)
    print(
        f"12: {optimal_ratio:.7f}   # Optimized ratio for 12th order "
        f"(±{abs(error):.2f} Hz)"
    )

    if verification["tolerance_met"]:
        print(
            "\n✅ Optimization successful! The 12th order filter now meets "
            "the 1.5 Hz tolerance requirement."
        )
    else:
        print(
            f"\n❌ Optimization did not meet the 1.5 Hz tolerance. "
            f"Best achieved: ±{abs(error):.2f} Hz"
        )
