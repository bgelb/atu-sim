from __future__ import annotations

from ..detectors import Detector
from ..lc_bank import LCBank
from .firmware import AlgoOptions, FirmwareAlgo
from .types import AlgoResult, TuningAlgo


class BGAlgo(TuningAlgo):
    """Improved BG search algorithm using the firmware-inspired state machine."""

    def __init__(
        self, bank: LCBank, detector: Detector, options: AlgoOptions | None = None
    ) -> None:
        self.impl = FirmwareAlgo(bank, detector, options)

    def run(self, freq_hz: float, z_load: complex) -> AlgoResult:
        return self.impl.run_bg(freq_hz, z_load)

    def name(self) -> str:
        return "bg"
