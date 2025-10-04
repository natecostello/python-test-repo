# pyyeti_minimal

Minimal extraction of pyyeti modules for Shock Response Spectrum (SRS) calculations.

## Contents

This package contains three modules extracted from [pyyeti](https://github.com/twmacro/pyyeti):

- **srs.py** - Shock Response Spectrum calculations
- **psd.py** - Power Spectral Density tools
- **dsp.py** - Digital Signal Processing tools

## Modifications

The original pyyeti dependencies have been stubbed out to make these modules standalone:

- `pyyeti.ytools._check_makeplot` → Stubbed in `dsp.py` (plotting functions disabled)
- Cross-module imports updated to use relative imports within this package

## Usage

```python
from pyyeti_minimal import srs
import numpy as np

# Create a signal
sr = 10000  # Sample rate (Hz)
duration = 0.01  # 10 ms
t = np.arange(0, duration, 1/sr)
pulse = 100 * np.sin(np.pi * t / duration)  # 100g half-sine pulse

# Calculate SRS
frequencies = np.array([10, 50, 100, 200, 500, 1000])  # Hz
Q = 10  # Quality factor (Q = 10 corresponds to 5% damping)
srs_result = srs.srs(pulse, sr, frequencies, Q)

print(f"SRS values: {srs_result.max(axis=0)}")
```

## Main Functions

### srs.srs()
Main SRS calculation function using time-domain integration.

### srs.srs_fft()
FFT-based SRS calculation (faster for many frequencies).

### srs.srsmap()
Calculate SRS over time windows (waterfall-style analysis).

## Notes

- Only features needed for SRS calculations are guaranteed to work
- Plotting functions in `dsp.py` are disabled (return None)
- This is a minimal extraction, not a full pyyeti replacement

## Original Source

These modules are adapted from [pyyeti](https://github.com/twmacro/pyyeti) by Tim Widrick.
All credit for the original implementation goes to the pyyeti project.
