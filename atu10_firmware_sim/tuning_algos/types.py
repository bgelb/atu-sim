from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol


class Topology(Enum):
    SHUNT_AT_LOAD = auto()
    SHUNT_AT_SOURCE = auto()


@dataclass
class TuningConfig:
    l_bits: int
    c_bits: int
    topology: Topology


@dataclass
class AlgoTrace:
    step: int
    phase: str
    topology: Topology
    l_bits: int
    c_bits: int
    z_in: complex
    detector_output: int


@dataclass
class AlgoResult:
    final_config: TuningConfig
    final_z_in: complex
    final_detector_output: int
    trace: list[AlgoTrace]

    @property
    def steps(self) -> int:
        return len(self.trace)


class TuningAlgo(Protocol):
    def run(self, freq_hz: float, z_load: complex) -> AlgoResult:  # pragma: no cover - interface
        ...

    def name(self) -> str:
        return self.__class__.__name__
