from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..detectors import Detector
from ..lc_bank import LCBank, ShuntPosition
from ..core import TunerSim, SimFlags, l_network_input_impedance, LCBank as LegacyBank
from .base import TuningAlgo, TuningConfig, Topology


def _topology_from_sw(sw: int) -> Topology:
    return Topology.SHUNT_AT_LOAD if sw == 0 else Topology.SHUNT_AT_SOURCE


def _shunt_pos_from_topology(topo: Topology) -> ShuntPosition:
    return ShuntPosition.LOAD if topo == Topology.SHUNT_AT_LOAD else ShuntPosition.SOURCE


@dataclass
class LegacyResult:
    final_config: TuningConfig
    trace: list[dict[str, Any]]
    steps: int
    final_swr: int


class LegacyBGAlgo(TuningAlgo):
    """
    Wrapper around existing BG algorithm to fit the new simulator interface.
    """

    def __init__(self, bank: LCBank) -> None:
        if len(bank.l_values) != 7 or len(bank.c_values) != 7:
            raise AssertionError("Legacy BG algo currently expects 7 L and 7 C relays")
        self.bank = bank
        self._legacy_bank = LegacyBank(l_values=bank.l_values, c_values=bank.c_values)
        self._config = TuningConfig(0, 1, Topology.SHUNT_AT_LOAD)
        self._done = False

    def reset(self) -> None:
        self._config = TuningConfig(0, 1, Topology.SHUNT_AT_LOAD)
        self._done = False

    def is_done(self) -> bool:
        return self._done

    def current_config(self) -> TuningConfig:
        return self._config

    def step(self, detector_output: int) -> None:
        # Legacy wrapper runs full tune in one shot; mark done
        self._done = True

    def run(self, freq_hz: float, z_load: complex, detector: Detector) -> LegacyResult:
        sim = TunerSim(
            freq_hz=freq_hz,
            z_load=z_load,
            bank=self._legacy_bank,
            flags=SimFlags(algorithm="bg", trace_steps=True),
        )
        sim.atu_reset()
        sim.tune()
        trace = []
        for t in sim.trace:
            topo = _topology_from_sw(t["SW"])
            z_in = l_network_input_impedance(
                freq_hz,
                z_load,
                t["ind"],
                t["cap"],
                t["SW"],
                sim.bank,
            )
            metric = detector.measure(z_in)
            trace.append(
                {
                    "step": t["step"],
                    "phase": t["phase"],
                    "topology": topo,
                    "l_bits": t["ind"],
                    "c_bits": t["cap"],
                    "z_in": z_in,
                    "swr": metric,
                }
            )

        final_cfg = TuningConfig(sim.ind, sim.cap, _topology_from_sw(sim.SW))
        return LegacyResult(
            final_config=final_cfg,
            trace=trace,
            steps=len(trace),
            final_swr=sim.SWR,
        )

    def name(self) -> str:
        return "bg"


class LegacyATU10Algo(LegacyBGAlgo):
    """
    Wrapper around reference ATU-10 firmware algorithm.
    """

    def __init__(self, bank: LCBank) -> None:
        super().__init__(bank)

    def run(self, freq_hz: float, z_load: complex, detector: Detector) -> LegacyResult:
        sim = TunerSim(
            freq_hz=freq_hz,
            z_load=z_load,
            bank=self._legacy_bank,
            flags=SimFlags(algorithm="atu10", trace_steps=True),
        )
        sim.atu_reset()
        sim.tune()
        trace = []
        for t in sim.trace:
            topo = _topology_from_sw(t["SW"])
            z_in = l_network_input_impedance(
                freq_hz,
                z_load,
                t["ind"],
                t["cap"],
                t["SW"],
                sim.bank,
            )
            metric = detector.measure(z_in)
            trace.append(
                {
                    "step": t["step"],
                    "phase": t["phase"],
                    "topology": topo,
                    "l_bits": t["ind"],
                    "c_bits": t["cap"],
                    "z_in": z_in,
                    "swr": metric,
                }
            )

        final_cfg = TuningConfig(sim.ind, sim.cap, _topology_from_sw(sim.SW))
        return LegacyResult(
            final_config=final_cfg,
            trace=trace,
            steps=len(trace),
            final_swr=sim.SWR,
        )

    def name(self) -> str:
        return "atu10"
