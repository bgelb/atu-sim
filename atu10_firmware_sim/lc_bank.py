from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math
from typing import Iterable


class ShuntPosition(Enum):
    LOAD = auto()    # shunt at load side (simulator SW=0)
    SOURCE = auto()  # shunt at source/input side (simulator SW=1)


@dataclass
class LCBank:
    l_values: tuple[float, ...]
    c_values: tuple[float, ...]

    def l_from_bits(self, bits: int) -> float:
        return sum(v for i, v in enumerate(self.l_values) if bits & (1 << i))

    def c_from_bits(self, bits: int) -> float:
        return sum(v for i, v in enumerate(self.c_values) if bits & (1 << i))

    def input_impedance(
        self,
        freq_hz: float,
        z_load: complex,
        l_bits: int,
        c_bits: int,
        shunt_pos: ShuntPosition,
    ) -> complex:
        w = 2 * math.pi * freq_hz
        L = self.l_from_bits(l_bits)
        C = self.c_from_bits(c_bits)

        if L == 0 and C == 0:
            return z_load

        j = 1j
        z_L_series = j * w * L if L > 0 else 0j
        y_C = j * w * C if C > 0 else 0j

        if shunt_pos == ShuntPosition.LOAD:
            y_load = 0j if abs(z_load) < 1e-12 else 1.0 / z_load
            y_total = y_load + y_C
            if abs(y_total) < 1e-18:
                z_node = complex(1e12, 0)
            else:
                z_node = 1.0 / y_total
            return z_L_series + z_node

        # Shunt at source/input side
        z_series = z_L_series + z_load
        y_series = 0j if abs(z_series) < 1e-12 else 1.0 / z_series
        y_in = y_series + y_C
        if abs(y_in) < 1e-18:
            return complex(1e12, 0)
        return 1.0 / y_in

    @staticmethod
    def _vswr_from_z(z_in: complex, z0: float = 50.0) -> float:
        if math.isinf(z_in.real) or math.isinf(z_in.imag):
            return math.inf

        denom = z_in + z0
        if abs(denom) < 1e-12:
            return math.inf

        gamma = (z_in - z0) / denom
        mag = abs(gamma)
        if mag >= 0.999999:
            return math.inf

        return (1.0 + mag) / (1.0 - mag)

    def get_swr(
        self,
        freq_hz: float,
        z_load: complex,
        l_bits: int,
        c_bits: int,
        shunt_pos: ShuntPosition,
        z0: float = 50.0,
    ) -> float:
        z_in = self.input_impedance(freq_hz, z_load, l_bits, c_bits, shunt_pos)
        return self._vswr_from_z(z_in, z0)

    def get_swr_map(
        self,
        freq_hz: float,
        z_load: complex,
        shunt_pos: ShuntPosition,
        z0: float = 50.0,
    ) -> list[list[float]]:
        l_range = range(1 << len(self.l_values))
        c_range = range(1 << len(self.c_values))
        grid: list[list[float]] = []
        for l_bits in l_range:
            row: list[float] = []
            for c_bits in c_range:
                row.append(self.get_swr(freq_hz, z_load, l_bits, c_bits, shunt_pos, z0))
            grid.append(row)
        return grid

    def get_best_swr(
        self,
        freq_hz: float,
        z_load: complex,
        z0: float = 50.0,
    ) -> tuple[float, int, int, ShuntPosition]:
        best_swr = math.inf
        best_state: tuple[int, int, ShuntPosition] | None = None
        for shunt_pos in (ShuntPosition.LOAD, ShuntPosition.SOURCE):
            grid = self.get_swr_map(freq_hz, z_load, shunt_pos, z0)
            for l_bits, row in enumerate(grid):
                for c_bits, swr_val in enumerate(row):
                    if swr_val < best_swr:
                        best_swr = swr_val
                        best_state = (l_bits, c_bits, shunt_pos)
        assert best_state is not None
        return best_swr, best_state[0], best_state[1], best_state[2]
