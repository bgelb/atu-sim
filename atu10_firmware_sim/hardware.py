from __future__ import annotations

from .lc_bank import LCBank


def atu10_bank() -> LCBank:
    """ATU-10 relay bank values."""
    return LCBank(
        l_values=(
            0.10e-6,
            0.22e-6,
            0.45e-6,
            1.0e-6,
            2.2e-6,
            4.5e-6,
            10.0e-6,
        ),
        c_values=(
            22e-12,
            47e-12,
            100e-12,
            220e-12,
            470e-12,
            1.0e-9,
            2.2e-9,
        ),
    )
