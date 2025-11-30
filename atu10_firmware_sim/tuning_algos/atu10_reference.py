from __future__ import annotations

from ..detectors import Detector
from ..lc_bank import LCBank
from .firmware import AlgoOptions, FirmwareAlgo
from .types import AlgoResult, TuningAlgo


class ATU10ReferenceAlgo(TuningAlgo):
    """Firmware-faithful ATU-10 algorithm."""

    def __init__(
        self, bank: LCBank, detector: Detector, options: AlgoOptions | None = None
    ) -> None:
        self.impl = FirmwareAlgo(bank, detector, options)

    def run(self, freq_hz: float, z_load: complex) -> AlgoResult:
        return self.impl.run_atu10(freq_hz, z_load)

    def name(self) -> str:
        return "atu10"
