from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class TuningConfig:
    l_bits: int
    c_bits: int
    topology: "Topology"


class Topology(Enum):
    SHUNT_AT_LOAD = auto()
    SHUNT_AT_SOURCE = auto()


class TuningAlgo:
    """Interface for tuning algorithms."""

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def is_done(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def current_config(self) -> TuningConfig:  # pragma: no cover
        raise NotImplementedError

    def step(self, detector_output: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
