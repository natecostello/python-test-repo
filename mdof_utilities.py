import numpy as np
from scipy.linalg import eig
from scipy.integrate import solve_ivp

"""
Multi-Degree-of-Freedom (MDOF) System Utilities

This module provides classes and functions for analyzing SDOF, 2-DOF, and 3-DOF systems
with base excitation, including transmissibility analysis and shock response spectrum
calculations.

Classes:
--------
- SDOF_System: Single degree of freedom system
- TDOF_System: Two degree of freedom system
- THREEDOF_System: Three degree of freedom system

Functions:
----------
- time_response: Compute MDOF system response to base acceleration time series
- shock_response_spectrum: Compute Shock Response Spectrum (SRS) with speed
  optimization options

Author: Generated from mdof_evaluation_refactored.ipynb
"""


# =============================================================================
# SYSTEM CLASSES
# =============================================================================


class SDOF_System:
    """Single degree of freedom system with base excitation."""

    def __init__(self, m, k, c):
        self.m, self.k, self.c = m, k, c
        self.fn = np.sqrt(k / m) / (2 * np.pi)  # Natural frequency (Hz)
        self.wn = np.sqrt(k / m)  # Natural frequency (rad/s)
        self.zeta = c / (2 * np.sqrt(k * m))  # Damping ratio

    def transmissibility(self, freq_range):
        """Calculate acceleration transmissibility vs frequency."""
        omega = 2 * np.pi * freq_range
        r = omega / self.wn  # Frequency ratio

        # Handle very low frequencies explicitly to avoid numerical issues
        trans = np.zeros_like(r)

        for i, r_val in enumerate(r):
            if r_val < 1e-6:  # Very low frequency - essentially static
                trans[i] = 1.0
            else:
                # Acceleration transmissibility: |X/Y| for base excitation
                # T = sqrt[(1 + (2*zeta*r)^2) / ((1 - r^2)^2 + (2*zeta*r)^2)]
                numerator = 1 + (2 * self.zeta * r_val) ** 2
                denominator = (1 - r_val**2) ** 2 + (2 * self.zeta * r_val) ** 2
                trans[i] = np.sqrt(numerator / denominator)

        return trans


class TDOF_System:
    """Two degree of freedom system with base excitation."""

    def __init__(self, m1, k1, c1, m2, k2, c2):
        self.m1, self.m2 = m1, m2
        self.k1, self.k2 = k1, k2
        self.c1, self.c2 = c1, c2

        # Mass and stiffness matrices
        self.M = np.array([[m1, 0], [0, m2]])
        self.K = np.array([[k1 + k2, -k2], [-k2, k2]])
        self.C = np.array([[c1 + c2, -c2], [-c2, c2]])

        # Calculate modal properties
        self._calculate_modal_properties()

    def _calculate_modal_properties(self):
        """Calculate natural frequencies and mode shapes."""
        eigenvalues, eigenvectors = eig(self.K, self.M)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.fn = np.sqrt(eigenvalues.real) / (2 * np.pi)  # Hz
        self.wn = np.sqrt(eigenvalues.real)  # rad/s
        self.mode_shapes = eigenvectors.real

    def transmissibility(self, freq_range):
        """Calculate acceleration transmissibility for both masses."""
        omega = 2 * np.pi * freq_range
        trans_m1 = np.zeros(len(omega))
        trans_m2 = np.zeros(len(omega))

        for i, w in enumerate(omega):
            if w < 1e-6:  # Very low frequency - essentially static
                trans_m1[i] = 1.0
                trans_m2[i] = 1.0
            else:
                # Dynamic stiffness matrix
                D = -(w**2) * self.M + 1j * w * self.C + self.K

                # Force vector for base excitation: F = M * ω² * 1
                # (unit base acceleration)
                F = w**2 * np.array([self.m1, self.m2])

                # Solve for absolute displacements relative to inertial frame
                # The equation is: D * X_rel = F where X_rel is relative to base
                # But we want absolute acceleration transmissibility
                # For base excitation with unit base acceleration:
                try:
                    X_rel = np.linalg.solve(D, F)
                    # Absolute displacement = base displacement + relative displacement
                    # For unit base displacement, absolute displacement = 1 + X_rel
                    X_abs = 1.0 + X_rel
                    # Acceleration transmissibility = |acceleration response|
                    # / |base acceleration| = |ω² * X_abs| / |ω² * 1| = |X_abs|
                    trans_m1[i] = np.abs(X_abs[0])
                    trans_m2[i] = np.abs(X_abs[1])
                except np.linalg.LinAlgError:
                    # Handle singular matrix (shouldn't happen for well-posed problems)
                    trans_m1[i] = 1.0
                    trans_m2[i] = 1.0

        return trans_m1, trans_m2


class THREEDOF_System:
    """Three degree of freedom system with base excitation."""

    def __init__(self, m1, k1, c1, m2, k2, c2, m3, k3, c3):
        self.m1, self.m2, self.m3 = m1, m2, m3
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.c1, self.c2, self.c3 = c1, c2, c3

        # Mass, stiffness, and damping matrices
        self.M = np.array([[m1, 0, 0], [0, m2, 0], [0, 0, m3]])
        self.K = np.array([[k1 + k2, -k2, 0], [-k2, k2 + k3, -k3], [0, -k3, k3]])
        self.C = np.array([[c1 + c2, -c2, 0], [-c2, c2 + c3, -c3], [0, -c3, c3]])

        # Calculate modal properties
        self._calculate_modal_properties()

    def _calculate_modal_properties(self):
        """Calculate natural frequencies and mode shapes."""
        eigenvalues, eigenvectors = eig(self.K, self.M)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.fn = np.sqrt(eigenvalues.real) / (2 * np.pi)  # Hz
        self.wn = np.sqrt(eigenvalues.real)  # rad/s
        self.mode_shapes = eigenvectors.real

    def transmissibility(self, freq_range):
        """Calculate acceleration transmissibility for all three masses."""
        omega = 2 * np.pi * freq_range
        trans_m1 = np.zeros(len(omega))
        trans_m2 = np.zeros(len(omega))
        trans_m3 = np.zeros(len(omega))

        for i, w in enumerate(omega):
            if w < 1e-6:  # Very low frequency - essentially static
                trans_m1[i] = 1.0
                trans_m2[i] = 1.0
                trans_m3[i] = 1.0
            else:
                # Dynamic stiffness matrix
                D = -(w**2) * self.M + 1j * w * self.C + self.K

                # Force vector for base excitation: F = M * ω² * 1
                # (unit base acceleration)
                F = w**2 * np.array([self.m1, self.m2, self.m3])

                # Solve for absolute displacements relative to inertial frame
                try:
                    X_rel = np.linalg.solve(D, F)
                    # Absolute displacement = base displacement + relative displacement
                    # For unit base displacement, absolute displacement = 1 + X_rel
                    X_abs = 1.0 + X_rel
                    # Acceleration transmissibility = |acceleration response|
                    # / |base acceleration| = |ω² * X_abs| / |ω² * 1| = |X_abs|
                    trans_m1[i] = np.abs(X_abs[0])
                    trans_m2[i] = np.abs(X_abs[1])
                    trans_m3[i] = np.abs(X_abs[2])
                except np.linalg.LinAlgError:
                    # Handle singular matrix (shouldn't happen for well-posed problems)
                    trans_m1[i] = 1.0
                    trans_m2[i] = 1.0
                    trans_m3[i] = 1.0

        return trans_m1, trans_m2, trans_m3


# =============================================================================
# TIME RESPONSE FUNCTION
# =============================================================================


def time_response(system, t, a_base):
    """
    Compute MDOF system response to base acceleration time series.

    Returns absolute accelerations of each mass.

    Parameters:
    -----------
    system : SDOF_System, TDOF_System, or THREEDOF_System
        The dynamic system to analyze
    t : array_like
        Time vector (seconds)
    a_base : array_like
        Base acceleration time series (g or m/s²)

    Returns:
    --------
    t_out : ndarray
        Output time vector (may be different from input due to solver)
    a_resp : ndarray
        Absolute acceleration response (shape: (n_dof, n_time))
        Each row corresponds to a mass response
    """
    # Build matrices
    if hasattr(system, "M") and hasattr(system, "C") and hasattr(system, "K"):
        M = np.atleast_2d(system.M)
        C = np.atleast_2d(system.C)
        K = np.atleast_2d(system.K)
    else:
        # SDOF fallback
        M = np.array([[system.m]])
        C = np.array([[system.c]])
        K = np.array([[system.k]])

    n = M.shape[0]  # number of DOF

    # Set up state-space: z = [x; xdot]
    def ode_fun(t_val, z):
        x = z[:n]
        xdot = z[n:]

        # Interpolate base acceleration
        a_b = np.interp(t_val, t, a_base)

        # xddot = M^-1 * (-C*xdot - K*x - M*a_base*1)
        rhs = -(C.dot(xdot) + K.dot(x)) - M.dot(np.ones(n)) * a_b
        xddot = np.linalg.solve(M, rhs)

        return np.concatenate([xdot, xddot])

    # Initial conditions: system at rest
    z0 = np.zeros(2 * n)

    # Solve ODE
    sol = solve_ivp(
        ode_fun,
        [t[0], t[-1]],
        z0,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        dense_output=True,
    )

    # Extract solution at original time points
    z = sol.sol(t)
    x = z[:n, :]
    xdot = z[n:, :]

    # Compute accelerations: xdd = M^-1[-C xdot - K x - M a_base(t)]
    a_b_vec = a_base  # already at time points t
    rhs = -(C.dot(xdot) + K.dot(x)) - M.dot(np.ones((n, len(a_b_vec)))) * a_b_vec
    xdd = np.linalg.solve(M, rhs)

    # Absolute acceleration = relative + base
    a_resp = xdd + a_b_vec[np.newaxis, :]

    return t, a_resp


def time_response_tdof(system, t, a_base, rtol=1e-7, atol=1e-9, method="RK45"):
    """
    Two-DOF base-excited stack, consistent with your TDOF_System indexing:
      index 0 (m1,k1,c1) = MIDDLE / base-connected
      index 1 (m2,k2,c2) = TOP (connected to middle via k2,c2)

    Coordinates integrated (pairwise-relative):
        u1 = x_top - x_mid        (across k2,c2)
        u2 = x_mid - y_base       (across k1,c1)

    EOM in these coordinates:
        m2*(u1¨ + u2¨) + c2*u1˙ + k2*u1               = - m2 * a_base
        m1*u2¨ + c1*u2˙ + k1*u2 - c2*u1˙ - k2*u1      = - m1 * a_base

    Absolute accelerations (RETURN ORDER MATCHES CLASS: [middle, top]):
        a_mid = u2¨ + a_base
        a_top = u1¨ + u2¨ + a_base
    """
    # Unpack
    m1 = float(system.m1)
    k1 = float(system.k1)
    c1 = float(system.c1)  # middle/base
    m2 = float(system.m2)
    k2 = float(system.k2)
    c2 = float(system.c2)  # top

    t = np.asarray(t, dtype=float)
    a_base = np.asarray(a_base, dtype=float)
    if t.ndim != 1 or a_base.ndim != 1 or len(t) != len(a_base):
        raise ValueError("t and a_base must be 1D arrays of equal length")

    # State z = [u1, u2, v1, v2]
    def rhs(ti, z):
        u1, u2, v1, v2 = z
        ab = np.interp(ti, t, a_base)

        # from EOM above
        r_upper = -c2 * v1 - k2 * u1 - m2 * ab
        r_lower = -m1 * ab - c1 * v2 - k1 * u2 + c2 * v1 + k2 * u1

        u2dd = r_lower / m1
        u1dd = (r_upper - m2 * u2dd) / m2

        return np.array([v1, v2, u1dd, u2dd], float)

    z0 = np.zeros(4)
    sol = solve_ivp(
        rhs, (t[0], t[-1]), z0, t_eval=t, rtol=rtol, atol=atol, method=method
    )

    # Recover accelerations exactly as in rhs (no finite differences)
    u1, u2, v1, v2 = sol.y
    ab = a_base
    r_upper = -c2 * v1 - k2 * u1 - m2 * ab
    r_lower = -m1 * ab - c1 * v2 - k1 * u2 + c2 * v1 + k2 * u1
    u2dd = r_lower / m1
    u1dd = (r_upper - m2 * u2dd) / m2

    a_mid = u2dd + ab  # row 0 (middle/base-connected)
    a_top = u1dd + u2dd + ab  # row 1 (top)

    return t, np.vstack((a_mid, a_top))


# =============================================================================
# SHOCK RESPONSE SPECTRUM FUNCTION
# =============================================================================


def shock_response_spectrum(
    t_input, a_input, freq_range, damping_ratio=0.05, speed_level="optimal"
):
    """
    Optimized Shock Response Spectrum (SRS) computation for rapid iteration.

    This function provides multiple speed levels to balance accuracy and
    computational time.

    Parameters:
    -----------
    t_input : array_like
        Time vector for input acceleration (seconds)
    a_input : array_like
        Input acceleration time series (g)
    freq_range : array_like
        Natural frequencies to evaluate SRS at (Hz)
    damping_ratio : float, optional
        Damping ratio for SDOF oscillators (default: 0.05 = 5%, Q=10)
    speed_level : str, optional
        Speed/accuracy trade-off level:
        - 'optimal': Best balance of speed & accuracy (RK45, robust)
        - 'high_order': High-order method (DOP853, sometimes less stable)
        - 'reference': Reference accuracy, same as original function

    Returns:
    --------
    srs_values : ndarray
        Maximum absolute acceleration response for each frequency (g)

    Notes:
    ------
    The 'optimal' mode uses RK45 integration which has been found to be more
    robust and often more accurate than high-order methods for SRS calculations.
    This is due to the nature of SRS involving many short, transient problems.
    """

    # Speed optimization parameters
    speed_params = {
        "optimal": {
            "rtol": 1e-4,  # Optimal tolerance for SRS problems
            "atol": 1e-6,  # Good balance of speed and accuracy
            "method": "RK45",  # Most robust method for SRS calculations
            "ringdown_factor": 2.0,  # Efficient ringdown time
            "min_points": 3,  # Minimum points per evaluation
        },
        "high_order": {
            "rtol": 1e-6,  # Tighter tolerance
            "atol": 1e-8,  # Higher precision
            "method": "DOP853",  # High-order method (can be less stable)
            "ringdown_factor": 3.0,  # Longer ringdown
            "min_points": 5,
        },
        "reference": {
            "rtol": 1e-8,  # Original tight tolerances
            "atol": 1e-10,  # Maximum precision
            "method": "DOP853",  # Same as original function
            "ringdown_factor": 5.0,  # Full ringdown time
            "min_points": 10,
        },
    }

    # Support legacy names for backward compatibility
    legacy_mapping = {
        "fast": "optimal",
        "medium": "high_order",
        "accurate": "reference",
    }

    # Map legacy names to new names
    if speed_level in legacy_mapping:
        print(
            f"⚠️  Legacy mode '{speed_level}' mapped to '{legacy_mapping[speed_level]}'"
        )
        speed_level = legacy_mapping[speed_level]

    params = speed_params.get(speed_level, speed_params["optimal"])

    srs_values = np.zeros(len(freq_range))

    # Ensure time starts at 0
    t_start = 0.0
    dt = t_input[1] - t_input[0]

    # Ring-down time based on speed level
    T_max = 1.0 / np.min(freq_range)
    t_ringdown = params["ringdown_factor"] * T_max

    # Adaptive ring-down points
    n_ringdown = max(int(t_ringdown / dt), params["min_points"])
    t_extended = np.concatenate(
        [t_input, t_input[-1] + dt * np.arange(1, n_ringdown + 1)]
    )
    a_extended = np.concatenate([a_input, np.zeros(n_ringdown)])

    # Pre-compute frequently used values
    freq_rad = 2 * np.pi * freq_range  # Convert to rad/s once

    for i, (fn, wn) in enumerate(zip(freq_range, freq_rad)):

        # SDOF equation for each frequency
        def sdof_ode(t, y):
            x, xdot = y

            # Fast linear interpolation
            if t <= t_extended[-1]:
                idx = np.searchsorted(t_extended, t)
                if idx == 0:
                    a_base = a_extended[0]
                elif idx >= len(t_extended):
                    a_base = 0.0
                else:
                    # Linear interpolation
                    t0, t1 = t_extended[idx - 1], t_extended[idx]
                    a0, a1 = a_extended[idx - 1], a_extended[idx]
                    a_base = a0 + (a1 - a0) * (t - t0) / (t1 - t0)
            else:
                a_base = 0.0

            # SDOF equation of motion: x'' + 2*zeta*wn*x' + wn^2*x = -a_base
            xddot = -2 * damping_ratio * wn * xdot - wn**2 * x - a_base
            return [xdot, xddot]

        # Initial conditions (system at rest)
        y0 = [0.0, 0.0]

        # Solve with optimized parameters
        try:
            sol = solve_ivp(
                sdof_ode,
                [t_start, t_extended[-1]],
                y0,
                method=params["method"],
                rtol=params["rtol"],
                atol=params["atol"],
                dense_output=False,
            )

            if sol.success:
                # Use solution points directly
                x = sol.y[0, :]  # relative displacement
                xdot = sol.y[1, :]  # relative velocity

                # Calculate relative acceleration at solution points
                a_base_interp = np.interp(sol.t, t_extended, a_extended)
                xddot = -2 * damping_ratio * wn * xdot - wn**2 * x - a_base_interp

                # Absolute acceleration = relative + base
                a_absolute = xddot + a_base_interp

                # SRS value is maximum absolute acceleration
                srs_values[i] = np.max(np.abs(a_absolute))

            else:
                print(f"Warning: Integration failed for frequency {fn:.1f} Hz")
                srs_values[i] = np.max(np.abs(a_input))

        except Exception as e:
            print(f"Error at frequency {fn:.1f} Hz: {str(e)}")
            srs_values[i] = np.max(np.abs(a_input))

    return srs_values
