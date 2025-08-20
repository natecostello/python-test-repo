import numpy as np
from scipy.linalg import eig
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete, lfilter

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
- generate_half_sine_pulse: Generate half-sine pulse shock input for testing
- synthesize_shock_pulse: Synthesize acceleration signals matching target SRS
  using wavelet or damped sine basis functions

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


# =============================================================================
# SHOCK INPUT GENERATION FUNCTIONS
# =============================================================================


def generate_half_sine_pulse(amplitude, duration, total_time, sample_rate=10000):
    """
    Generate a half-sine pulse shock input.

    This is the classic shock input used in Steinberg's analysis and
    military/aerospace shock testing standards.

    Parameters:
    -----------
    amplitude : float
        Peak acceleration amplitude (g)
    duration : float
        Pulse duration (seconds)
    total_time : float
        Total analysis time (seconds)
    sample_rate : float
        Sampling rate (Hz)

    Returns:
    --------
    t : ndarray
        Time vector
    a : ndarray
        Acceleration time series
    """
    dt = 1.0 / sample_rate
    t = np.arange(0, total_time, dt)
    a = np.zeros_like(t)

    # Half-sine pulse during pulse duration
    pulse_mask = t <= duration
    a[pulse_mask] = amplitude * np.sin(np.pi * t[pulse_mask] / duration)

    return t, a


# =============================================================================
# SHOCK SYNTHESIS FUNCTIONS
# =============================================================================


def synthesize_shock_pulse(
    srs_spec_hz_g,
    fs=20480,
    duration=0.25,
    q=10.0,
    freqs_per_octave=12,
    n_trials=120,
    inner_iters=18,
    nm_choices=(5, 7, 9, 11, 13),
    rng_seed=None,
    # shock-shaping knobs (both bases):
    t0=0.010,  # main shock start [s]
    tail_span=0.060,  # length of trailing tail [s]
    focus=0.85,  # 0–1: how tightly arrivals cluster near t0
    late_energy_tau=0.050,  # window for the time-concentration score [s]
    w_time=0.6,  # weight of time-concentration in winner score
    w_simplicity=0.08,  # weight on zero-crossing count
    clip_scale=(0.25, 4.0),  # numerical guard for SRS scaling (not a physical limit)
    # basis selection:
    basis="wavelet",  # "wavelet" or "damped_sine"
    ds_zeta=0.06,  # damping ratio for damped-sine basis (≈ 1/(2Q_ds))
    zero_drift_fix=None,  # None | "poly" (optional drift cleanup for damped_sine)
):
    """
    SRS-matching shock synthesis with a 'shocky' time profile.

    Synthesizes acceleration time series that match a specified Shock Response Spectrum
    (SRS) using either compact-support NESC wavelets or damped sine basis functions
    with iterative SRS scaling optimization.

    Choose basis="wavelet" (compact-support NESC wavelets) or basis="damped_sine"
    (A * e^{-ζ*2πf*t} * sin(2πf*t)), both using iterative SRS scaling.

    Parameters
    ----------
    srs_spec_hz_g : array_like
        Target SRS specification as (M,2) array of (freq_Hz, SRS_G) pairs
    fs : float, optional
        Sample rate (Hz), default 20480
    duration : float, optional
        Signal duration (seconds), default 0.25
    q : float, optional
        Quality factor for SRS calculation, default 10.0
    freqs_per_octave : int, optional
        Frequency resolution for synthesis grid, default 12
    n_trials : int, optional
        Number of random synthesis trials, default 120
    inner_iters : int, optional
        SRS scaling iterations per trial, default 18
    nm_choices : tuple, optional
        Wavelet support parameter choices, default (5,7,9,11,13)
    rng_seed : int, optional
        Random seed for reproducibility, default None
    t0 : float, optional
        Main shock start time (s), default 0.010
    tail_span : float, optional
        Length of trailing tail (s), default 0.060
    focus : float, optional
        Clustering tightness near t0 (0-1), default 0.85
    late_energy_tau : float, optional
        Time window for concentration score (s), default 0.050
    w_time : float, optional
        Weight of time-concentration in score, default 0.6
    w_simplicity : float, optional
        Weight of zero-crossing count, default 0.08
    clip_scale : tuple, optional
        Scaling guard limits, default (0.25, 4.0)
    basis : str, optional
        Basis function type: "wavelet" or "damped_sine", default "wavelet"
    ds_zeta : float, optional
        Damping ratio for damped-sine basis, default 0.06
    zero_drift_fix : str, optional
        Drift cleanup method: None or "poly", default None

    Returns
    -------
    t : ndarray
        Time vector (seconds)
    acc_g : ndarray
        Synthesized acceleration time series (g)
    report : dict
        Synthesis results and metrics including:
        - freqs_hz: frequency grid used
        - target_srs_g: target SRS values
        - achieved_srs_g: achieved SRS values
        - max_abs_error_db: maximum SRS error (dB)
        - peak_accel_g: peak acceleration (g)
        - winner_trial: best trial number
        - basis: basis function used

    Examples
    --------
    >>> # Define target SRS specification
    >>> srs_spec = np.array([[10.0, 20.0],    # 10 Hz -> 20g
    ...                      [100.0, 50.0],   # 100 Hz -> 50g
    ...                      [1000.0, 50.0]]) # 1000 Hz -> 50g
    >>>
    >>> # Synthesize with damped sine basis
    >>> t, acc, info = synthesize_shock_pulse(srs_spec,
    ...                                       basis="damped_sine",
    ...                                       n_trials=100)
    >>>
    >>> print(f"Peak acceleration: {info['peak_accel_g']:.1f} g")
    >>> print(f"SRS matching error: {info['max_abs_error_db']:.2f} dB")

    Notes
    -----
    The algorithm uses a two-level optimization:
    1. Outer loop: Random trials with different delay patterns and parameters
    2. Inner loop: Iterative SRS-based amplitude scaling for each trial

    Damped sine basis often performs better with lower complexity settings
    (fewer freqs_per_octave, fewer inner_iters) due to natural physics matching
    and overfitting prevention.

    References
    ----------
    Based on shock synthesis methods for vibration testing and SRS matching.
    """

    rng = np.random.default_rng(rng_seed)

    # ---------- Helper functions ----------
    def log_interp(x, xp, fp):
        """Logarithmic interpolation"""
        x = np.asarray(x)
        xp = np.asarray(xp)
        fp = np.asarray(fp)
        lx = np.log10(x)
        lxp = np.log10(xp)
        lfp = np.log10(fp)
        out = np.interp(lx, lxp, lfp, left=lfp[0], right=lfp[-1])
        return 10**out

    def build_freq_grid(f_lo, f_hi, nper_oct):
        """Build logarithmic frequency grid"""
        n_oct = np.log2(f_hi / f_lo)
        n_pts = int(np.floor(n_oct * nper_oct)) + 1
        return f_lo * (2.0 ** (np.arange(n_pts) / nper_oct))

    def basis_row_wavelet(t, A, f, delay, Nm):
        """Compact-support (odd Nm>=5) wavelet: zero net v/disp for each component."""
        start = delay
        stop = delay + Nm * 0.5 / f
        idx = (t >= start) & (t <= stop)
        z = np.zeros_like(t)
        tau = t[idx] - delay
        z[idx] = A * np.sin(2 * np.pi * f * tau / Nm) * np.sin(2 * np.pi * f * tau)
        return z

    def basis_row_damped_sine(t, A, f, delay, zeta):
        """A * exp(-zeta*2π f (t-d)) * sin(2π f (t-d)) for t>=delay; 0 otherwise."""
        idx = t >= delay
        z = np.zeros_like(t)
        tau = t[idx] - delay
        z[idx] = A * np.exp(-zeta * 2 * np.pi * f * tau) * np.sin(2 * np.pi * f * tau)
        return z

    def srs_accel_abs(acc_g, fs, freqs_hz, q):
        """Absolute-acceleration SRS via bilinear-discretized SDOF filters."""
        G2SI = 9.80665
        ydd = np.asarray(acc_g) * G2SI
        zeta = 1.0 / (2.0 * q)
        dt = 1.0 / fs
        out = np.empty_like(freqs_hz, float)
        for i, fn in enumerate(freqs_hz):
            wn = 2 * np.pi * fn
            num = [2 * zeta * wn, wn**2]
            den = [1.0, 2 * zeta * wn, wn**2]
            bz, az, _ = cont2discrete((num, den), dt, method="bilinear")
            b = bz.flatten()
            a = az.flatten()
            a_abs = lfilter(b, a, ydd)
            out[i] = np.max(np.abs(a_abs)) / G2SI
        return out

    def clustered_delays(freqs, support_like, duration, t0, tail_span, focus, rng):
        """High-f earlier (near t0), low-f later; small jitter ∝ support."""
        order = np.argsort(-freqs)  # descending f -> ranks 0..1
        r = np.empty_like(freqs, float)
        r[order] = np.linspace(0.0, 1.0, freqs.size)
        base = t0 + r * tail_span
        jitter = (rng.random(freqs.size) - 0.5) * (1.0 - focus) * support_like
        delay = np.clip(
            base + jitter, 0.0, np.maximum(0.0, duration - support_like - 2.0 / fs)
        )
        return delay

    def zero_crossings(x):
        """Count zero crossings in signal"""
        s = np.signbit(x)
        return int(np.count_nonzero(s[1:] ^ s[:-1]))

    # ---------- Input validation and setup ----------
    srs_spec_hz_g = np.asarray(srs_spec_hz_g, float)
    if srs_spec_hz_g.ndim != 2 or srs_spec_hz_g.shape[1] != 2:
        raise ValueError("srs_spec_hz_g must be (M,2) array of (freq_Hz, SRS_G).")
    fmin, fmax = float(srs_spec_hz_g[0, 0]), float(srs_spec_hz_g[-1, 0])
    freqs = build_freq_grid(fmin, fmax, freqs_per_octave)
    target_srs = log_interp(freqs, srs_spec_hz_g[:, 0], srs_spec_hz_g[:, 1])

    t = np.arange(int(round(duration * fs))) / fs

    # Basis selector
    use_wavelet = basis.lower() == "wavelet"
    if not use_wavelet and basis.lower() != "damped_sine":
        raise ValueError("basis must be 'wavelet' or 'damped_sine'.")

    # ---------- Main synthesis loop (outer trials) ----------
    best = dict(score=np.inf)
    for trial in range(n_trials):
        if use_wavelet:
            Nm = np.asarray(
                [
                    np.random.default_rng(rng.integers(1 << 31)).choice(nm_choices)
                    for _ in freqs
                ]
            )
            # support length for delay placement
            support_like = Nm * 0.5 / freqs
        else:
            Nm = None
            # effective support proxy for placing delays (few decay constants)
            support_like = np.minimum(duration, 3.0 / (ds_zeta * 2 * np.pi * freqs))

        # start with positive amplitudes to favor a single dominant lobe
        A = 0.3 * target_srs * (1 + 0.5 * (rng.random(freqs.size) - 0.5))
        A = np.abs(A)

        delay = clustered_delays(
            freqs, support_like, duration, t0, tail_span, focus, rng
        )

        # precompute basis rows (unit amplitude)
        basis_rows = np.empty((freqs.size, t.size), float)
        if use_wavelet:
            for k, (fk, dk, Nk) in enumerate(zip(freqs, delay, Nm)):
                basis_rows[k] = basis_row_wavelet(t, 1.0, fk, dk, Nk)
        else:
            for k, (fk, dk) in enumerate(zip(freqs, delay)):
                basis_rows[k] = basis_row_damped_sine(t, 1.0, fk, dk, ds_zeta)

        # Inner loop: SRS amplitude scaling
        for _ in range(inner_iters):
            acc = A @ basis_rows
            achieved = srs_accel_abs(acc, fs, freqs, q)
            scale = np.divide(target_srs, np.maximum(achieved, 1e-12))
            scale = np.clip(scale, *clip_scale)  # stability only
            A *= scale

        # Final evaluation for this trial
        acc = A @ basis_rows

        # Optional drift cleanup for damped_sine (analysis convenience)
        if (not use_wavelet) and (zero_drift_fix == "poly"):
            # remove tiny linear trend in velocity to suppress residual drift
            G2SI = 9.80665
            v = np.cumsum(acc * G2SI) / fs
            # fit a line to v and subtract its derivative from acceleration
            n = len(v)
            x = np.arange(n)
            c1, c0 = np.polyfit(x, v, 1)  # v ≈ c1*x + c0
            acc = acc - (c1 * G2SI) / fs  # derivative of linear term

        achieved = srs_accel_abs(acc, fs, freqs, q)
        err_db = 20.0 * np.log10(
            np.maximum(achieved, 1e-12) / np.maximum(target_srs, 1e-12)
        )
        max_abs_err_db = float(np.max(np.abs(err_db)))

        # Shock-aware score (analysis mode): SRS error + time-concentration + simplicity
        win_lo = int(np.floor(t0 * fs))
        win_hi = int(np.floor(min(duration, t0 + late_energy_tau) * fs))
        e_total = float(np.sum(acc**2) + 1e-18)
        e_focus = float(np.sum(acc[win_lo:win_hi] ** 2))
        time_penalty = 1.0 - (e_focus / e_total)  # smaller is better
        zc = zero_crossings(acc)
        simplicity_penalty = zc / max(1, len(acc) // 8)
        score = (
            max_abs_err_db
            + w_time * (10.0 * time_penalty)
            + w_simplicity * (10.0 * simplicity_penalty)
        )

        if score < best.get("score", np.inf) or (
            np.isclose(score, best.get("score", np.inf))
            and max_abs_err_db < best.get("max_abs_err_db", np.inf)
        ):
            G2SI = 9.80665
            v = np.cumsum(acc * G2SI) / fs
            d = np.cumsum(v) / fs
            best.update(
                dict(
                    score=score,
                    max_abs_err_db=max_abs_err_db,
                    acc=acc.copy(),
                    achieved=achieved.copy(),
                    trial=trial,
                    time_focus_ratio=e_focus / e_total,
                    zero_crossings=int(zc),
                    peakA=float(np.max(np.abs(acc))),
                    peakV=float(np.max(np.abs(v))),
                    peakD=float(np.max(np.abs(d))),
                    delays=delay.copy(),
                    Nm=None if Nm is None else Nm.copy(),
                    basis=basis.lower(),
                )
            )

    # Prepare final report
    report = dict(
        freqs_hz=freqs,
        target_srs_g=target_srs,
        achieved_srs_g=best["achieved"],
        max_abs_error_db=best["max_abs_err_db"],
        time_focus_ratio=float(best["time_focus_ratio"]),
        zero_crossings=int(best["zero_crossings"]),
        peak_accel_g=float(best["peakA"]),
        peak_velocity_si=float(best["peakV"]),
        peak_displacement_si=float(best["peakD"]),
        winner_trial=int(best["trial"]),
        q=float(q),
        fs=float(fs),
        t0=float(t0),
        tail_span=float(tail_span),
        basis=best["basis"],
    )
    return t, best["acc"], report
