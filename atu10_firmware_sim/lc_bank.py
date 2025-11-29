from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math


class ShuntPosition(Enum):
    LOAD = auto()    # shunt at load side (simulator SW=0)
    SOURCE = auto()  # shunt at source/input side (simulator SW=1)


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

    def swr(
        self,
        freq_hz: float,
        z_load: complex,
        l_bits: int,
        c_bits: int,
        shunt_pos: ShuntPosition,
        z0: float = 50.0,
    ) -> int:
        z_in = self.input_impedance(freq_hz, z_load, l_bits, c_bits, shunt_pos)
        return swr_from_z(z_in, z0)
