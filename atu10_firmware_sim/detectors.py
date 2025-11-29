from __future__ import annotations

from dataclasses import dataclass

from .lc_bank import swr_from_z


class Detector:
    """Interface for detectors converting impedance to a feedback metric."""

    def measure(self, z_in: complex) -> int:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class ATU10IntegerVSWRDetector(Detector):
    z0: float = 50.0

    def measure(self, z_in: complex) -> int:
        return swr_from_z(z_in, self.z0)
