"""
Airfoil self-noise prediction with polynomial regression.

A small, reproducible workflow for the MMA3001 individual project. It predicts
the scaled sound pressure level (SPL, dB) of trailing-edge self-noise from a
NACA 0012 airfoil section, using the NASA wind-tunnel data published by
Brooks, Pope and Marcolini (1989) and hosted by the UCI Machine Learning
Repository (dataset id 291).

Prediction contract
-------------------
- **Prediction unit (unit of observation):** one one-third-octave frequency band
  from one wind-tunnel test run.
- **Target:** scaled SPL in that band (dB).
- **Available inputs:** frequency (Hz), angle of attack (deg), chord length (m),
  free-stream velocity (m/s), suction-side displacement thickness (m). All are
  known before the noise is measured.
- **Deployment population:** untested designs or operating conditions *within*
  the tested range (NACA 0012, chord 0.0254-0.3048 m, 31.7-71.3 m/s).
- **Decision and cost:** early-stage noise screening of blade sections.
  Underestimating noise risks a design that later fails noise limits;
  overestimating leads to over-conservative designs.

Modules
-------
- `airfoil_noise.data` - loading, auditing, run grouping and input checks.
- `airfoil_noise.models` - linear baseline and polynomial pipelines.
- `airfoil_noise.evaluation` - metrics, random vs grouped CV, degree sweep,
  timing and the physical (velocity) trend check.
- `airfoil_noise.plots` - figures used in the report.
"""

__version__ = "0.1.0"
