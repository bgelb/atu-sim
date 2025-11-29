from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .detectors import Detector
from .lc_bank import LCBank, ShuntPosition, swr_from_z
from .tuning_algos.base import TuningConfig, Topology
from .tuning_algos.legacy import LegacyATU10Algo, LegacyBGAlgo, LegacyResult


@dataclass
class TuneTraceEntry:
    step: int
    topology: Topology
    sw: int
    l_bits: int
    c_bits: int
    z_in: complex
    swr: int
    detector_output: int
    phase: str


@dataclass
class TuneResult:
    final_config: TuningConfig
    final_z_in: complex
    final_swr: int
    detector_output: int
    steps: int
    trace: list[TuneTraceEntry]


class ATUSimulator:
    """
    Orchestrates a tune using an LCBank, Detector, and TuningAlgo.

    Currently wraps legacy algorithms; future algos can implement the base
    TuningAlgo interface directly.
    """

    def __init__(
        self,
        bank: LCBank,
        detector: Detector,
        algorithm: str = "bg",
        z0: float = 50.0,
    ) -> None:
        self.bank = bank
        self.detector = detector
        self.z0 = z0
        algo_l = algorithm.lower()
        if algo_l == "bg":
            self.algo = LegacyBGAlgo(bank)
        elif algo_l == "atu10":
            self.algo = LegacyATU10Algo(bank)
        else:
            raise ValueError(f"Unknown algorithm {algorithm}")

    def tune(self, freq_hz: float, z_load: complex) -> TuneResult:
        # For now the legacy algo runs the full tune internally
        if isinstance(self.algo, (LegacyBGAlgo, LegacyATU10Algo)):
            legacy_result: LegacyResult = self.algo.run(freq_hz, z_load, self.detector)
            trace_entries: list[TuneTraceEntry] = []
            for t in legacy_result.trace:
                topo = t["topology"]
                z_in = t["z_in"]
                metric = self.detector.measure(z_in)
                trace_entries.append(
                    TuneTraceEntry(
                        step=t["step"],
                        topology=topo,
                        sw=0 if topo == Topology.SHUNT_AT_LOAD else 1,
                        l_bits=t["l_bits"],
                        c_bits=t["c_bits"],
                        z_in=z_in,
                        swr=metric,
                        detector_output=metric,
                        phase=t["phase"],
                    )
                )
            final_topo = legacy_result.final_config.topology
            z_in_final = self.bank.input_impedance(
                freq_hz,
                z_load,
                legacy_result.final_config.l_bits,
                legacy_result.final_config.c_bits,
                ShuntPosition.LOAD
                if final_topo == Topology.SHUNT_AT_LOAD
                else ShuntPosition.SOURCE,
            )
            detector_out = self.detector.measure(z_in_final)
            return TuneResult(
                final_config=legacy_result.final_config,
                final_z_in=z_in_final,
                final_swr=legacy_result.final_swr,
                detector_output=detector_out,
                steps=legacy_result.steps,
                trace=trace_entries,
            )

        raise NotImplementedError("Non-legacy algos not yet wired")
