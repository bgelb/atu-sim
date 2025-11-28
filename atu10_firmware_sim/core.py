from __future__ import annotations

from dataclasses import dataclass, field
import math

Z0_DEFAULT = 50.0


@dataclass
class SimFlags:
    """
    Optional knobs to experiment with algorithm tweaks while keeping defaults
    faithful to the firmware when unset.

    force_all_coarse_strategies:
        If True, always run coarse strategies 2 and 3 regardless of the
        cap/ind gate. Leave False for firmware-faithful behavior.
    """

    force_all_coarse_strategies: bool = False


@dataclass
class LCBank:
    l_values: tuple[float, ...] = (
        0.10e-6,
        0.22e-6,
        0.45e-6,
        1.0e-6,
        2.2e-6,
        4.5e-6,
        10.0e-6,
    )
    c_values: tuple[float, ...] = (
        22e-12,
        47e-12,
        100e-12,
        220e-12,
        470e-12,
        1.0e-9,
        2.2e-9,
    )

    def l_from_bits(self, bits: int) -> float:
        return sum(v for i, v in enumerate(self.l_values) if bits & (1 << i))

    def c_from_bits(self, bits: int) -> float:
        return sum(v for i, v in enumerate(self.c_values) if bits & (1 << i))

    def nearest_lc(
        self, l_target: float, c_target: float
    ) -> tuple[int, float, int, float, bool]:
        """
        Find relay bitmasks that most closely realize the requested L and C.

        Returns (l_bits, l_henry, c_bits, c_farad, in_range) where in_range
        is True only if both requested targets fall within achievable min/max.
        """
        l_bits_best, l_value_best = min(
            ((bits, self.l_from_bits(bits)) for bits in range(1 << len(self.l_values))),
            key=lambda pair: abs(pair[1] - l_target),
        )
        c_bits_best, c_value_best = min(
            ((bits, self.c_from_bits(bits)) for bits in range(1 << len(self.c_values))),
            key=lambda pair: abs(pair[1] - c_target),
        )

        l_max = self.l_from_bits((1 << len(self.l_values)) - 1)
        c_max = self.c_from_bits((1 << len(self.c_values)) - 1)
        in_range = (0.0 <= l_target <= l_max) and (0.0 <= c_target <= c_max)

        return l_bits_best, l_value_best, c_bits_best, c_value_best, in_range


def _safe_inv(z: complex) -> complex:
    if abs(z) < 1e-12:
        return 0j
    return 1.0 / z


def l_network_input_impedance(
    freq_hz: float,
    z_load: complex,
    l_bits: int,
    c_bits: int,
    sw: int,
    bank: LCBank | None = None,
) -> complex:
    if bank is None:
        bank = LCBank()

    w = 2 * math.pi * freq_hz
    L = bank.l_from_bits(l_bits)
    C = bank.c_from_bits(c_bits)

    if L == 0 and C == 0:
        return z_load

    j = 1j
    z_L_series = j * w * L if L > 0 else 0j
    y_C = j * w * C if C > 0 else 0j

    if sw == 0:
        y_load = _safe_inv(z_load)
        y_total = y_load + y_C
        if abs(y_total) < 1e-18:
            z_node = complex(1e12, 0)
        else:
            z_node = 1.0 / y_total
        return z_L_series + z_node
    else:
        z_series = z_L_series + z_load
        y_series = _safe_inv(z_series)
        y_in = y_series + y_C
        if abs(y_in) < 1e-18:
            return complex(1e12, 0)
        return 1.0 / y_in


def swr_from_z(z_in: complex, z0: float = Z0_DEFAULT) -> int:
    if math.isinf(z_in.real) or math.isinf(z_in.imag):
        return 999

    if abs(z_in + z0) < 1e-12:
        return 999

    gamma = (z_in - z0) / (z_in + z0)
    mag = abs(gamma)
    if mag >= 0.999:
        return 999

    s = (1.0 + mag) / (1.0 - mag)
    if s > 9.985:
        return 999

    swr_int = int(s * 100.0 + 0.5)
    return max(swr_int, 100)


@dataclass
class TunerSim:
    freq_hz: float
    z_load: complex
    bank: LCBank = field(default_factory=LCBank)
    z0: float = Z0_DEFAULT
    flags: SimFlags = field(default_factory=SimFlags)

    ind: int = 0
    cap: int = 0
    SW: int = 0
    SWR: int = 999

    def atu_reset(self) -> None:
        self.ind = 0
        self.cap = 1
        self.SW = 0
        self.get_swr()

    def get_swr(self) -> None:
        z_in = l_network_input_impedance(
            self.freq_hz, self.z_load, self.ind, self.cap, self.SW, self.bank
        )
        self.SWR = swr_from_z(z_in, self.z0)

    def relay_set(
        self,
        ind: int | None = None,
        cap: int | None = None,
        SW: int | None = None,
    ) -> None:
        if ind is not None:
            self.ind = ind
        if cap is not None:
            self.cap = cap
        if SW is not None:
            self.SW = SW
        self.get_swr()

    def coarse_cap(self) -> None:
        cap_mem = 0
        self.get_swr()
        swr_mem = self.SWR // 10

        cap = 1
        while cap < 64:
            self.relay_set(cap=cap)
            swr_scaled = self.SWR // 10
            if swr_scaled <= swr_mem:
                cap_mem = cap
                swr_mem = swr_scaled
                cap *= 2
            else:
                break

        self.cap = cap_mem
        self.relay_set(cap=self.cap)

    def coarse_ind(self) -> None:
        ind_mem = 0
        self.get_swr()
        swr_mem = self.SWR // 10

        ind = 1
        while ind < 64:
            self.relay_set(ind=ind)
            swr_scaled = self.SWR // 10
            if swr_scaled <= swr_mem:
                ind_mem = ind
                swr_mem = swr_scaled
                ind *= 2
            else:
                break

        self.ind = ind_mem
        self.relay_set(ind=self.ind)

    def coarse_ind_cap(self) -> None:
        ind_mem = 0
        self.get_swr()
        swr_mem = self.SWR // 10

        ind = 1
        while ind < 64:
            self.relay_set(ind=ind, cap=ind)
            swr_scaled = self.SWR // 10
            if swr_scaled <= swr_mem:
                ind_mem = ind
                swr_mem = swr_scaled
                ind *= 2
            else:
                break

        self.ind = ind_mem
        self.cap = ind_mem
        self.relay_set(ind=self.ind, cap=self.cap)

    def coarse_tune(self) -> None:
        SWR_mem1 = 10000
        SWR_mem2 = 10000
        SWR_mem3 = 10000
        ind_mem1 = cap_mem1 = 0
        ind_mem2 = cap_mem2 = 0
        ind_mem3 = cap_mem3 = 0

        self.coarse_cap()
        self.coarse_ind()
        self.get_swr()
        if self.SWR <= 120:
            return

        SWR_mem1 = self.SWR
        ind_mem1 = self.ind
        cap_mem1 = self.cap

        allow_alt = self.cap <= 2 and self.ind <= 2
        if self.flags.force_all_coarse_strategies:
            allow_alt = True

        if allow_alt:
            self.ind = 0
            self.cap = 0
            self.relay_set()
            self.coarse_ind()
            self.coarse_cap()
            self.get_swr()
            if self.SWR <= 120:
                return
            SWR_mem2 = self.SWR
            ind_mem2 = self.ind
            cap_mem2 = self.cap

        if allow_alt:
            self.ind = 0
            self.cap = 0
            self.relay_set()
            self.coarse_ind_cap()
            self.get_swr()
            if self.SWR <= 120:
                return
            SWR_mem3 = self.SWR
            ind_mem3 = self.ind
            cap_mem3 = self.cap

        if SWR_mem1 <= SWR_mem2 and SWR_mem1 <= SWR_mem3:
            self.cap = cap_mem1
            self.ind = ind_mem1
        elif SWR_mem2 <= SWR_mem1 and SWR_mem2 <= SWR_mem3:
            self.cap = cap_mem2
            self.ind = ind_mem2
        elif SWR_mem3 <= SWR_mem1 and SWR_mem3 <= SWR_mem2:
            self.cap = cap_mem3
            self.ind = ind_mem3

        self.relay_set()

    def sharp_cap(self) -> None:
        cap_mem = self.cap
        step = self.cap // 10
        if step == 0:
            step = 1

        self.get_swr()
        swr_mem = self.SWR

        cap_trial = self.cap + step
        if cap_trial > 127:
            cap_trial = 127
        self.relay_set(cap=cap_trial)

        if self.SWR <= swr_mem:
            swr_mem = self.SWR
            cap_mem = self.cap
            while True:
                cap_trial = self.cap + step
                if cap_trial > (127 - step):
                    break
                self.relay_set(cap=cap_trial)
                if self.SWR <= swr_mem:
                    cap_mem = self.cap
                    swr_mem = self.SWR
                    step = self.cap // 10
                    if step == 0:
                        step = 1
                else:
                    break
        else:
            swr_mem = self.SWR
            while True:
                cap_trial = self.cap - step
                if cap_trial < step:
                    break
                self.relay_set(cap=cap_trial)
                if self.SWR <= swr_mem:
                    cap_mem = self.cap
                    swr_mem = self.SWR
                    step = self.cap // 10
                    if step == 0:
                        step = 1
                else:
                    break

        self.relay_set(cap=cap_mem)

    def sharp_ind(self) -> None:
        ind_mem = self.ind
        step = self.ind // 10
        if step == 0:
            step = 1

        self.get_swr()
        swr_mem = self.SWR

        ind_trial = self.ind + step
        if ind_trial > 127:
            ind_trial = 127
        self.relay_set(ind=ind_trial)

        if self.SWR <= swr_mem:
            swr_mem = self.SWR
            ind_mem = self.ind
            while True:
                ind_trial = self.ind + step
                if ind_trial > (127 - step):
                    break
                self.relay_set(ind=ind_trial)
                if self.SWR <= swr_mem:
                    ind_mem = self.ind
                    swr_mem = self.SWR
                    step = self.ind // 10
                    if step == 0:
                        step = 1
                else:
                    break
        else:
            swr_mem = self.SWR
            while True:
                ind_trial = self.ind - step
                if ind_trial < step:
                    break
                self.relay_set(ind=ind_trial)
                if self.SWR <= swr_mem:
                    ind_mem = self.ind
                    swr_mem = self.SWR
                    step = self.ind // 10
                    if step == 0:
                        step = 1
                else:
                    break

        self.relay_set(ind=ind_mem)

    def sharp_tune(self) -> None:
        if self.cap >= self.ind:
            self.sharp_cap()
            self.sharp_ind()
        else:
            self.sharp_ind()
            self.sharp_cap()

    def subtune(self) -> None:
        self.ind = 0
        self.cap = 0
        self.relay_set()
        if self.SWR <= 120:
            return

        self.coarse_tune()
        self.get_swr()
        if self.SWR <= 120:
            return

        self.sharp_tune()

    def tune(self) -> None:
        self.get_swr()
        if self.SWR <= 120:
            return

        self.subtune()
        self.get_swr()
        if self.SWR <= 120:
            return

        SWR_mem = self.SWR
        cap_mem = self.cap
        ind_mem = self.ind

        self.SW = 0 if self.SW == 1 else 1
        self.subtune()
        self.get_swr()

        if self.SWR > SWR_mem:
            self.SW = 0 if self.SW == 1 else 1
            self.ind = ind_mem
            self.cap = cap_mem
            self.relay_set()
            self.get_swr()

        if self.SWR <= 120:
            return

        self.sharp_tune()
        self.get_swr()

        if self.SWR == 999:
            self.atu_reset()


def brute_force_best(
    freq_hz: float,
    z_load: complex,
    bank: LCBank | None = None,
) -> tuple[int, tuple[int, int, int]]:
    if bank is None:
        bank = LCBank()

    best_swr = 999
    best_state: tuple[int, int, int] | None = None

    for sw in (0, 1):
        for ind in range(0, 128):
            for cap in range(0, 128):
                z_in = l_network_input_impedance(freq_hz, z_load, ind, cap, sw, bank)
                swr = swr_from_z(z_in)
                if swr < best_swr:
                    best_swr = swr
                    best_state = (ind, cap, sw)

    assert best_state is not None
    return best_swr, best_state
