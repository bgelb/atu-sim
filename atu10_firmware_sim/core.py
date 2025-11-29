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

    debug_coarse:
        If True, print coarse-tune step-by-step details (for investigation).

    trace_steps:
        If True, record every relay-set step (phase, SW, ind, cap, SWR) for
        later plotting or analysis.

    algorithm:
        Select tuning algorithm. "bg" (default) is the improved search
        described in the README; "atu10" preserves the firmware-faithful
        behavior.
    """

    force_all_coarse_strategies: bool = False
    debug_coarse: bool = False
    trace_steps: bool = False
    algorithm: str = "bg"


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
        #22.0e-6,
    )
    # ATU10 values
    c_values: tuple[float, ...] = (
        22e-12,
        47e-12,
        100e-12,
        220e-12,
        470e-12,
        1.0e-9,
        2.2e-9,
    )
    # T1 values
    #c_values: tuple[float, ...] = (
    #    10e-12,
    #    20e-12,
    #    39e-12,
    #    82e-12,
    #    160e-12,
    #    330e-12,
    #    660e-12,
    #)
    

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
    trace: list[dict] = field(default_factory=list)
    _trace_step: int = 0

    def _fmt_swr_val(self, swr: int) -> str:
        if swr >= 999:
            return ">= 9.99"
        return f"{swr / 100.0:.2f}"

    def _dbg(self, msg: str) -> None:
        if self.flags.debug_coarse:
            print(msg)

    def _trace(self, phase: str) -> None:
        if not self.flags.trace_steps:
            return
        self._trace_step += 1
        self.trace.append(
            {
                "step": self._trace_step,
                "phase": phase,
                "SW": self.SW,
                "ind": self.ind,
                "cap": self.cap,
                "SWR": self.SWR,
            }
        )

    def atu_reset(self) -> None:
        self.ind = 0
        self.cap = 1
        self.SW = 0
        self.get_swr()
        self._trace("reset")

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
        phase: str | None = None,
    ) -> None:
        if ind is not None:
            self.ind = ind
        if cap is not None:
            self.cap = cap
        if SW is not None:
            self.SW = SW
        self.get_swr()
        if phase:
            self._trace(phase)

    def coarse_cap(self) -> None:
        cap_mem = 0
        self.get_swr()
        swr_mem = self.SWR // 10

        cap = 1
        while cap < 64:
            self.relay_set(cap=cap, phase="coarse_cap")
            swr_scaled = self.SWR // 10
            self._dbg(
                f"        cap step: cap={cap:3d} SWR={self._fmt_swr_val(self.SWR)} scaled={swr_scaled}"
            )
            if swr_scaled <= swr_mem:
                cap_mem = cap
                swr_mem = swr_scaled
                cap *= 2
            else:
                break

        self.cap = cap_mem
        self.relay_set(cap=self.cap, phase="coarse_cap_final")
        self._dbg(
            f"        coarse_cap chosen cap={self.cap:3d} SWR={self._fmt_swr_val(self.SWR)}"
        )

    def coarse_ind(self) -> None:
        ind_mem = 0
        self.get_swr()
        swr_mem = self.SWR // 10

        ind = 1
        while ind < 64:
            self.relay_set(ind=ind, phase="coarse_ind")
            swr_scaled = self.SWR // 10
            self._dbg(
                f"        ind step: ind={ind:3d} SWR={self._fmt_swr_val(self.SWR)} scaled={swr_scaled}"
            )
            if swr_scaled <= swr_mem:
                ind_mem = ind
                swr_mem = swr_scaled
                ind *= 2
            else:
                break

        self.ind = ind_mem
        self.relay_set(ind=self.ind, phase="coarse_ind_final")
        self._dbg(
            f"        coarse_ind chosen ind={self.ind:3d} SWR={self._fmt_swr_val(self.SWR)}"
        )

    def coarse_ind_cap(self) -> None:
        ind_mem = 0
        self.get_swr()
        swr_mem = self.SWR // 10

        ind = 1
        while ind < 64:
            self.relay_set(ind=ind, cap=ind, phase="coarse_ind_cap")
            swr_scaled = self.SWR // 10
            self._dbg(
                f"        ind=cap step: val={ind:3d} SWR={self._fmt_swr_val(self.SWR)} scaled={swr_scaled}"
            )
            if swr_scaled <= swr_mem:
                ind_mem = ind
                swr_mem = swr_scaled
                ind *= 2
            else:
                break

        self.ind = ind_mem
        self.cap = ind_mem
        self.relay_set(ind=self.ind, cap=self.cap, phase="coarse_ind_cap_final")
        self._dbg(
            f"        coarse_ind_cap chosen ind=cap={self.ind:3d} SWR={self._fmt_swr_val(self.SWR)}"
        )

    def coarse_tune(self) -> None:
        SWR_mem1 = 10000
        SWR_mem2 = 10000
        SWR_mem3 = 10000
        ind_mem1 = cap_mem1 = 0
        ind_mem2 = cap_mem2 = 0
        ind_mem3 = cap_mem3 = 0

        debug_results: list[tuple[str, int, int, int]] = []

        self._dbg(
            f"    coarse_tune start SW={self.SW} ind={self.ind:3d} cap={self.cap:3d} SWR={self._fmt_swr_val(self.SWR)}"
        )
        self._trace("coarse_tune_start")

        self._dbg("      Strategy 1 (coarse_cap -> coarse_ind):")
        self.coarse_cap()
        self.coarse_ind()
        self.get_swr()
        if self.SWR <= 120:
            if self.flags.debug_coarse:
                debug_results.append(("Strategy 1", self.ind, self.cap, self.SWR))
            return

        SWR_mem1 = self.SWR
        ind_mem1 = self.ind
        cap_mem1 = self.cap
        if self.flags.debug_coarse:
            debug_results.append(("Strategy 1", self.ind, self.cap, self.SWR))

        allow_alt = self.cap <= 2 and self.ind <= 2
        if self.flags.force_all_coarse_strategies:
            allow_alt = True

        if allow_alt:
            self._dbg("      Strategy 2 (coarse_ind -> coarse_cap):")
            self.ind = 0
            self.cap = 0
            self.relay_set(phase="coarse_strategy2_reset")
            self.coarse_ind()
            self.coarse_cap()
            self.get_swr()
            if self.SWR <= 120:
                if self.flags.debug_coarse:
                    debug_results.append(
                        ("Strategy 2", self.ind, self.cap, self.SWR)
                    )
                return
            SWR_mem2 = self.SWR
            ind_mem2 = self.ind
            cap_mem2 = self.cap
            if self.flags.debug_coarse:
                debug_results.append(("Strategy 2", self.ind, self.cap, self.SWR))
        else:
            self._dbg("      Strategy 2 skipped (cap>2 or ind>2 after Strategy 1)")
            SWR_mem2 = 10000

        if allow_alt:
            self._dbg("      Strategy 3 (coarse_ind_cap):")
            self.ind = 0
            self.cap = 0
            self.relay_set(phase="coarse_strategy3_reset")
            self.coarse_ind_cap()
            self.get_swr()
            if self.SWR <= 120:
                if self.flags.debug_coarse:
                    debug_results.append(
                        ("Strategy 3", self.ind, self.cap, self.SWR)
                    )
                return
            SWR_mem3 = self.SWR
            ind_mem3 = self.ind
            cap_mem3 = self.cap
            if self.flags.debug_coarse:
                debug_results.append(("Strategy 3", self.ind, self.cap, self.SWR))
        else:
            self._dbg("      Strategy 3 skipped (cap>2 or ind>2 after Strategy 1)")
            SWR_mem3 = 10000

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

        if self.flags.debug_coarse and debug_results:
            self._dbg("      Summary:")
            for name, ind_v, cap_v, swr_v in debug_results:
                self._dbg(
                    f"        {name}: ind={ind_v:3d} cap={cap_v:3d} SWR={self._fmt_swr_val(swr_v)}"
                )

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
        self.relay_set(cap=cap_trial, phase="sharp_cap")

        if self.SWR <= swr_mem:
            swr_mem = self.SWR
            cap_mem = self.cap
            while True:
                cap_trial = self.cap + step
                if cap_trial > (127 - step):
                    break
                self.relay_set(cap=cap_trial, phase="sharp_cap")
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
                self.relay_set(cap=cap_trial, phase="sharp_cap")
                if self.SWR <= swr_mem:
                    cap_mem = self.cap
                    swr_mem = self.SWR
                    step = self.cap // 10
                    if step == 0:
                        step = 1
                else:
                    break

        self.relay_set(cap=cap_mem, phase="sharp_cap_final")

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
        self.relay_set(ind=ind_trial, phase="sharp_ind")

        if self.SWR <= swr_mem:
            swr_mem = self.SWR
            ind_mem = self.ind
            while True:
                ind_trial = self.ind + step
                if ind_trial > (127 - step):
                    break
                self.relay_set(ind=ind_trial, phase="sharp_ind")
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
                self.relay_set(ind=ind_trial, phase="sharp_ind")
                if self.SWR <= swr_mem:
                    ind_mem = self.ind
                    swr_mem = self.SWR
                    step = self.ind // 10
                    if step == 0:
                        step = 1
                else:
                    break

        self.relay_set(ind=ind_mem, phase="sharp_ind_final")

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
        self.relay_set(phase="subtune_reset")
        if self.SWR <= 120:
            return

        self.coarse_tune()
        self.get_swr()
        if self.SWR <= 120:
            return

        self.sharp_tune()

    def tune(self) -> None:
        if self.flags.algorithm == "atu10":
            return self._tune_atu10()
        return self._tune_bg()

    def _tune_atu10(self) -> None:
        if self.flags.trace_steps:
            self.trace.clear()
            self._trace_step = 0

        self.get_swr()
        self._trace("tune_start")
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
        self._trace("toggle_sw")
        self.subtune()
        self.get_swr()

        if self.SWR > SWR_mem:
            self.SW = 0 if self.SW == 1 else 1
            self.ind = ind_mem
            self.cap = cap_mem
            self.relay_set(phase="restore_state")
            self.get_swr()

        if self.SWR <= 120:
            return

        self.sharp_tune()
        self.get_swr()
        self._trace("tune_end")

        if self.SWR == 999:
            self.atu_reset()

    def _bg_apply_state(self, sw: int, primary: int, secondary: int, phase: str) -> None:
        """
        Helper to set relays with a primary/secondary axis abstraction.
        For sw=0: primary=L(ind), secondary=C(cap).
        For sw=1: primary=C(cap), secondary=L(ind).
        """
        if sw == 0:
            self.relay_set(ind=primary, cap=secondary, SW=sw, phase=phase)
        else:
            self.relay_set(ind=secondary, cap=primary, SW=sw, phase=phase)

    def _bg_eval_swr(self, sw: int, primary: int, secondary: int) -> int:
        if sw == 0:
            z_in = l_network_input_impedance(self.freq_hz, self.z_load, primary, secondary, sw, self.bank)
        else:
            z_in = l_network_input_impedance(self.freq_hz, self.z_load, secondary, primary, sw, self.bank)
        return swr_from_z(z_in, self.z0)

    def _bg_walk_primary(
        self, sw: int, primary_start: int, secondary: int, best_swr: int, phase_prefix: str
    ) -> tuple[int, int]:
        """
        Walk primary axis (L for sw=0, C for sw=1) up/down by 1 until SWR worsens by >0.2.
        Returns (best_primary, best_swr).
        """
        best_primary = primary_start
        best_val = best_swr

        # Increase direction
        p = primary_start
        while p < 127:
            p += 1
            self._bg_apply_state(sw, p, secondary, f"{phase_prefix}_inc")
            swr = self.SWR
            if swr < best_val:
                best_val = swr
                best_primary = p
            elif swr > best_val + 20:
                break

        # Decrease direction
        p = primary_start
        while p > 0:
            p -= 1
            self._bg_apply_state(sw, p, secondary, f"{phase_prefix}_dec")
            swr = self.SWR
            if swr < best_val:
                best_val = swr
                best_primary = p
            elif swr > best_val + 20:
                break

        self._bg_apply_state(sw, best_primary, secondary, f"{phase_prefix}_best")
        return best_primary, best_val

    def _bg_search_secondary(
        self, sw: int, primary_best: int, secondary_best: int, best_swr: int
    ) -> tuple[int, int, int]:
        """
        Explore neighboring secondary values with primary walks.
        """
        offset = 1
        global_best = best_swr
        best_p = primary_best
        best_s = secondary_best

        while True:
            improved = False
            for sign in (-1, 1):
                s_candidate = secondary_best + sign * offset
                if s_candidate < 0 or s_candidate > 127:
                    continue

                # Start from best primary found so far
                self._bg_apply_state(sw, best_p, s_candidate, "bg_sec_start")
                start_swr = self.SWR
                p_start = best_p

                if start_swr >= 900:
                    found = False
                    for radius in range(2, 128, 2):
                        candidates = []
                        if p_start - radius >= 0:
                            candidates.append(p_start - radius)
                        if p_start + radius <= 127:
                            candidates.append(p_start + radius)
                        best_trial: tuple[int, int] | None = None
                        for p in candidates:
                            self._bg_apply_state(sw, p, s_candidate, "bg_sec_probe")
                            swr_try = self.SWR
                            if swr_try < 900:
                                if best_trial is None or swr_try < best_trial[1]:
                                    best_trial = (p, swr_try)
                                found = True
                        if found and best_trial is not None:
                            p_start = best_trial[0]
                            self._bg_apply_state(sw, p_start, s_candidate, "bg_sec_seed")
                            break

                p_best_local, swr_local = self._bg_walk_primary(
                    sw, p_start, s_candidate, self.SWR, "bg_sec_walk"
                )
                if swr_local < global_best:
                    global_best = swr_local
                    best_p = p_best_local
                    best_s = s_candidate
                    improved = True

            if not improved:
                break
            secondary_best = best_s
            primary_best = best_p
            offset += 1

        self._bg_apply_state(sw, best_p, best_s, "bg_final_best")
        return best_p, best_s, global_best

    def _tune_bg(self) -> None:
        # Clear trace if needed
        if self.flags.trace_steps:
            self.trace.clear()
            self._trace_step = 0

        self.get_swr()
        self._trace("bg_start")

        ind_candidates = [0, 15, 31, 63, 95, 111, 127]
        cap_candidates = [0, 1, 2, 3, 5, 7, 9, 11, 15, 19, 23, 27, 43, 59, 75, 91]

        best_overall: tuple[int, int, int] | None = None  # (swr, primary, secondary, sw)
        best_sw = 0
        best_primary = 0
        best_secondary = 0
        best_swr = 999

        for sw in (0, 1):
            for ind in ind_candidates:
                for cap in cap_candidates:
                    self._bg_apply_state(sw, ind if sw == 0 else cap, cap if sw == 0 else ind, "bg_coarse_eval")
                    swr = self.SWR
                    if swr < best_swr:
                        best_swr = swr
                        best_sw = sw
                        best_primary = ind if sw == 0 else cap
                        best_secondary = cap if sw == 0 else ind
                        best_overall = (swr, best_primary, best_secondary, best_sw)
        if best_overall is None:
            self.atu_reset()
            return

        # Apply best coarse state
        self._bg_apply_state(best_sw, best_primary, best_secondary, "bg_coarse_best")
        best_swr = self.SWR

        # Primary walk
        best_primary, best_swr = self._bg_walk_primary(
            best_sw, best_primary, best_secondary, best_swr, "bg_primary"
        )

        # Secondary exploration with primary walks
        best_primary, best_secondary, best_swr = self._bg_search_secondary(
            best_sw, best_primary, best_secondary, best_swr
        )

        # Final state already applied in _bg_search_secondary
        self._trace("bg_end")


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


def swr_grid(
    freq_hz: float,
    z_load: complex,
    bank: LCBank | None = None,
) -> dict[int, list[list[int]]]:
    """
    Compute SWR for every relay combination (128x128) for both topologies.

    Returns a dict keyed by SW (0 or 1) with a 128x128 list of SWR ints.
    Rows are inductor bitmasks 0..127; columns are capacitor bitmasks 0..127.
    """
    if bank is None:
        bank = LCBank()

    grids: dict[int, list[list[int]]] = {0: [], 1: []}

    for sw in (0, 1):
        grid: list[list[int]] = []
        for ind in range(128):
            row: list[int] = []
            for cap in range(128):
                z_in = l_network_input_impedance(freq_hz, z_load, ind, cap, sw, bank)
                swr = swr_from_z(z_in)
                row.append(swr)
            grid.append(row)
        grids[sw] = grid

    return grids
