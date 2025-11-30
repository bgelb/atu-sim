from __future__ import annotations

from dataclasses import dataclass
import math


class Detector:
    """Interface for detectors converting impedance to a feedback metric."""

    def measure(self, z_in: complex) -> int:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class ATU10IntegerVSWRDetector(Detector):
    z0: float = 50.0

    @staticmethod
    def swr_from_z(z_in: complex, z0: float = 50.0) -> int:
        if math.isinf(z_in.real) or math.isinf(z_in.imag):
            return 999

        denom = z_in + z0
        if abs(denom) < 1e-12:
            return 999

        gamma = (z_in - z0) / denom
        mag = abs(gamma)
        if mag >= 0.999:
            return 999

        s = (1.0 + mag) / (1.0 - mag)
        if s > 9.985:
            return 999

        swr_int = int(s * 100.0 + 0.5)
        return max(swr_int, 100)

    def measure(self, z_in: complex) -> int:
        return self.swr_from_z(z_in, self.z0)
