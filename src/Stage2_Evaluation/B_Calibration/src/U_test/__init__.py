"""
U-test perturbation calibration utilities.

Core functions extracted from the SLURM driver so they can be shared between
`Slurm_version/U-test_perturbation_calibration/U-test_perturbation_calibration.py`
and `JupterNote_Version/U-test_perturbation_calibration.ipynb`:

- run real / fake (calibration null) perturbation association tests
- reload pre-computed results (auto-discovering sample names)
- plot violin (-ln(p) real vs null) and QQ diagnostics
"""

from .calibration import (
    compute_real_perturbation_tests,
    compute_fake_perturbation_tests,
    load_real_perturbation_tests,
    load_fake_perturbation_tests,
    plot_calibration_comparison,
    plot_qq_comparison,
)

__all__ = [
    "compute_real_perturbation_tests",
    "compute_fake_perturbation_tests",
    "load_real_perturbation_tests",
    "load_fake_perturbation_tests",
    "plot_calibration_comparison",
    "plot_qq_comparison",
]
