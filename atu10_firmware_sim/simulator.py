from __future__ import annotations

from dataclasses import dataclass
from .tuning_algos.types import AlgoResult, TuningAlgo, TuningConfig, Topology


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
    """

    def __init__(
        self,
        algorithm: TuningAlgo,
    ) -> None:
        self.algo = algorithm

    def tune(self, freq_hz: float, z_load: complex) -> TuneResult:
        algo_result: AlgoResult = self.algo.run(freq_hz, z_load)
        trace_entries: list[TuneTraceEntry] = []
        for t in algo_result.trace:
            trace_entries.append(
                TuneTraceEntry(
                    step=t.step,
                    topology=t.topology,
                    sw=0 if t.topology == Topology.SHUNT_AT_LOAD else 1,
                    l_bits=t.l_bits,
                    c_bits=t.c_bits,
                    z_in=t.z_in,
                    swr=t.detector_output,
                    detector_output=t.detector_output,
                    phase=t.phase,
                )
            )

        final_z_in = algo_result.final_z_in
        detector_out = algo_result.final_detector_output

        return TuneResult(
            final_config=algo_result.final_config,
            final_z_in=final_z_in,
            final_swr=detector_out,
            detector_output=detector_out,
            steps=len(trace_entries),
            trace=trace_entries,
        )
