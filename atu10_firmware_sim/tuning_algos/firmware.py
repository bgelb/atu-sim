from __future__ import annotations

from dataclasses import dataclass

from ..detectors import Detector
from ..lc_bank import LCBank, ShuntPosition
from .types import AlgoResult, AlgoTrace, Topology, TuningConfig


def _topology_from_sw(sw: int) -> Topology:
    return Topology.SHUNT_AT_LOAD if sw == 0 else Topology.SHUNT_AT_SOURCE


@dataclass
class AlgoOptions:
    force_all_coarse_strategies: bool = False
    debug_coarse: bool = False
    trace_steps: bool = True


class FirmwareAlgo:
    """
    Implements the reference ATU-10 firmware state machine (both the stock
    algorithm and the BG improved search) using the new LCBank/Detector
    interfaces.
    """

    def __init__(
        self,
        bank: LCBank,
        detector: Detector,
        options: AlgoOptions | None = None,
    ) -> None:
        if len(bank.l_values) != 7 or len(bank.c_values) != 7:
            raise AssertionError("FirmwareAlgo expects 7 L relays and 7 C relays")
        self.bank = bank
        self.detector = detector
        self.options = options or AlgoOptions()
        self.trace: list[AlgoTrace] = []
        self._trace_step = 0
        self.freq_hz = 0.0
        self.z_load: complex = 0j
        self.ind = 0
        self.cap = 0
        self.sw = 0
        self.swr = 999
        self.z_in = complex(0, 0)

    # ---- generic helpers ----
    def _dbg(self, msg: str) -> None:
        if self.options.debug_coarse:
            print(msg)

    def _update_measurement(self) -> None:
        pos = ShuntPosition.LOAD if self.sw == 0 else ShuntPosition.SOURCE
        self.z_in = self.bank.input_impedance(
            self.freq_hz, self.z_load, self.ind, self.cap, pos
        )
        self.swr = self.detector.measure(self.z_in)

    def _trace(self, phase: str) -> None:
        if not self.options.trace_steps:
            return
        self._trace_step += 1
        self.trace.append(
            AlgoTrace(
                step=self._trace_step,
                phase=phase,
                topology=_topology_from_sw(self.sw),
                l_bits=self.ind,
                c_bits=self.cap,
                z_in=self.z_in,
                detector_output=self.swr,
            )
        )

    def _set_state(
        self,
        ind: int | None = None,
        cap: int | None = None,
        sw: int | None = None,
        phase: str | None = None,
    ) -> None:
        if ind is not None:
            self.ind = ind
        if cap is not None:
            self.cap = cap
        if sw is not None:
            self.sw = sw
        self._update_measurement()
        if phase:
            self._trace(phase)

    def _prepare_run(self, freq_hz: float, z_load: complex) -> None:
        self.freq_hz = freq_hz
        self.z_load = z_load
        self.trace.clear()
        self._trace_step = 0
        self.atu_reset()

    def _result(self) -> AlgoResult:
        final_config = TuningConfig(
            l_bits=self.ind,
            c_bits=self.cap,
            topology=_topology_from_sw(self.sw),
        )
        return AlgoResult(
            final_config=final_config,
            final_z_in=self.z_in,
            final_detector_output=self.swr,
            trace=list(self.trace),
        )

    # ---- reused ATU-10 firmware logic ----
    def atu_reset(self) -> None:
        self.ind = 0
        self.cap = 1
        self.sw = 0
        self._update_measurement()
        self._trace("reset")

    def coarse_cap(self) -> None:
        cap_mem = 0
        self._update_measurement()
        swr_mem = self.swr // 10

        cap = 1
        while cap < 64:
            self._set_state(cap=cap, phase="coarse_cap")
            swr_scaled = self.swr // 10
            self._dbg(
                f"        cap step: cap={cap:3d} SWR={self.swr} scaled={swr_scaled}"
            )
            if swr_scaled <= swr_mem:
                cap_mem = cap
                swr_mem = swr_scaled
                cap *= 2
            else:
                break

        self.cap = cap_mem
        self._set_state(cap=self.cap, phase="coarse_cap_final")
        self._dbg(
            f"        coarse_cap chosen cap={self.cap:3d} SWR={self.swr}"
        )

    def coarse_ind(self) -> None:
        ind_mem = 0
        self._update_measurement()
        swr_mem = self.swr // 10

        ind = 1
        while ind < 64:
            self._set_state(ind=ind, phase="coarse_ind")
            swr_scaled = self.swr // 10
            self._dbg(
                f"        ind step: ind={ind:3d} SWR={self.swr} scaled={swr_scaled}"
            )
            if swr_scaled <= swr_mem:
                ind_mem = ind
                swr_mem = swr_scaled
                ind *= 2
            else:
                break

        self.ind = ind_mem
        self._set_state(ind=self.ind, phase="coarse_ind_final")
        self._dbg(
            f"        coarse_ind chosen ind={self.ind:3d} SWR={self.swr}"
        )

    def coarse_ind_cap(self) -> None:
        ind_mem = 0
        self._update_measurement()
        swr_mem = self.swr // 10

        ind = 1
        while ind < 64:
            self._set_state(ind=ind, cap=ind, phase="coarse_ind_cap")
            swr_scaled = self.swr // 10
            self._dbg(
                f"        ind=cap step: val={ind:3d} SWR={self.swr} scaled={swr_scaled}"
            )
            if swr_scaled <= swr_mem:
                ind_mem = ind
                swr_mem = swr_scaled
                ind *= 2
            else:
                break

        self.ind = ind_mem
        self.cap = ind_mem
        self._set_state(ind=self.ind, cap=self.cap, phase="coarse_ind_cap_final")
        self._dbg(
            f"        coarse_ind_cap chosen ind=cap={self.ind:3d} SWR={self.swr}"
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
            f"    coarse_tune start SW={self.sw} ind={self.ind:3d} cap={self.cap:3d} SWR={self.swr}"
        )
        self._trace("coarse_tune_start")

        self._dbg("      Strategy 1 (coarse_cap -> coarse_ind):")
        self.coarse_cap()
        self.coarse_ind()
        self._update_measurement()
        if self.swr <= 120:
            if self.options.debug_coarse:
                debug_results.append(("Strategy 1", self.ind, self.cap, self.swr))
            return

        SWR_mem1 = self.swr
        ind_mem1 = self.ind
        cap_mem1 = self.cap
        if self.options.debug_coarse:
            debug_results.append(("Strategy 1", self.ind, self.cap, self.swr))

        allow_alt = self.cap <= 2 and self.ind <= 2
        if self.options.force_all_coarse_strategies:
            allow_alt = True

        if allow_alt:
            self._dbg("      Strategy 2 (coarse_ind -> coarse_cap):")
            self.ind = 0
            self.cap = 0
            self._set_state(phase="coarse_strategy2_reset")
            self.coarse_ind()
            self.coarse_cap()
            self._update_measurement()
            if self.swr <= 120:
                if self.options.debug_coarse:
                    debug_results.append(
                        ("Strategy 2", self.ind, self.cap, self.swr)
                    )
                return
            SWR_mem2 = self.swr
            ind_mem2 = self.ind
            cap_mem2 = self.cap
            if self.options.debug_coarse:
                debug_results.append(("Strategy 2", self.ind, self.cap, self.swr))
        else:
            self._dbg("      Strategy 2 skipped (cap>2 or ind>2 after Strategy 1)")
            SWR_mem2 = 10000

        if allow_alt:
            self._dbg("      Strategy 3 (coarse_ind_cap):")
            self.ind = 0
            self.cap = 0
            self._set_state(phase="coarse_strategy3_reset")
            self.coarse_ind_cap()
            self._update_measurement()
            if self.swr <= 120:
                if self.options.debug_coarse:
                    debug_results.append(
                        ("Strategy 3", self.ind, self.cap, self.swr)
                    )
                return
            SWR_mem3 = self.swr
            ind_mem3 = self.ind
            cap_mem3 = self.cap
            if self.options.debug_coarse:
                debug_results.append(("Strategy 3", self.ind, self.cap, self.swr))
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

        self._set_state()

        if self.options.debug_coarse and debug_results:
            self._dbg("      Summary:")
            for name, ind_v, cap_v, swr_v in debug_results:
                self._dbg(
                    f"        {name}: ind={ind_v:3d} cap={cap_v:3d} SWR={swr_v}"
                )

    def sharp_cap(self) -> None:
        cap_mem = self.cap
        step = self.cap // 10
        if step == 0:
            step = 1

        self._update_measurement()
        swr_mem = self.swr

        cap_trial = self.cap + step
        if cap_trial > 127:
            cap_trial = 127
        self._set_state(cap=cap_trial, phase="sharp_cap")

        if self.swr <= swr_mem:
            swr_mem = self.swr
            cap_mem = self.cap
            while True:
                cap_trial = self.cap + step
                if cap_trial > (127 - step):
                    break
                self._set_state(cap=cap_trial, phase="sharp_cap")
                if self.swr <= swr_mem:
                    cap_mem = self.cap
                    swr_mem = self.swr
                    step = self.cap // 10
                    if step == 0:
                        step = 1
                else:
                    break
        else:
            swr_mem = self.swr
            while True:
                cap_trial = self.cap - step
                if cap_trial < step:
                    break
                self._set_state(cap=cap_trial, phase="sharp_cap")
                if self.swr <= swr_mem:
                    cap_mem = self.cap
                    swr_mem = self.swr
                    step = self.cap // 10
                    if step == 0:
                        step = 1
                else:
                    break

        self._set_state(cap=cap_mem, phase="sharp_cap_final")

    def sharp_ind(self) -> None:
        ind_mem = self.ind
        step = self.ind // 10
        if step == 0:
            step = 1

        self._update_measurement()
        swr_mem = self.swr

        ind_trial = self.ind + step
        if ind_trial > 127:
            ind_trial = 127
        self._set_state(ind=ind_trial, phase="sharp_ind")

        if self.swr <= swr_mem:
            swr_mem = self.swr
            ind_mem = self.ind
            while True:
                ind_trial = self.ind + step
                if ind_trial > (127 - step):
                    break
                self._set_state(ind=ind_trial, phase="sharp_ind")
                if self.swr <= swr_mem:
                    ind_mem = self.ind
                    swr_mem = self.swr
                    step = self.ind // 10
                    if step == 0:
                        step = 1
                else:
                    break
        else:
            swr_mem = self.swr
            while True:
                ind_trial = self.ind - step
                if ind_trial < step:
                    break
                self._set_state(ind=ind_trial, phase="sharp_ind")
                if self.swr <= swr_mem:
                    ind_mem = self.ind
                    swr_mem = self.swr
                    step = self.ind // 10
                    if step == 0:
                        step = 1
                else:
                    break

        self._set_state(ind=ind_mem, phase="sharp_ind_final")

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
        self._set_state(phase="subtune_reset")
        if self.swr <= 120:
            return

        self.coarse_tune()
        self._update_measurement()
        if self.swr <= 120:
            return

        self.sharp_tune()

    def _tune_atu10(self) -> None:
        if self.options.trace_steps:
            self.trace.clear()
            self._trace_step = 0

        self._update_measurement()
        self._trace("tune_start")
        if self.swr <= 120:
            return

        self.subtune()
        self._update_measurement()
        if self.swr <= 120:
            return

        SWR_mem = self.swr
        cap_mem = self.cap
        ind_mem = self.ind

        self.sw = 0 if self.sw == 1 else 1
        self._trace("toggle_sw")
        self.subtune()
        self._update_measurement()

        if self.swr > SWR_mem:
            self.sw = 0 if self.sw == 1 else 1
            self.ind = ind_mem
            self.cap = cap_mem
            self._set_state(phase="restore_state")
            self._update_measurement()

        if self.swr <= 120:
            return

        self.sharp_tune()
        self._update_measurement()
        self._trace("tune_end")

        if self.swr == 999:
            self.atu_reset()

    def _bg_apply_state(self, sw: int, primary: int, secondary: int, phase: str) -> None:
        """
        Helper to set relays with a primary/secondary axis abstraction.
        For sw=0: primary=L(ind), secondary=C(cap).
        For sw=1: primary=C(cap), secondary=L(ind).
        """
        if sw == 0:
            self._set_state(ind=primary, cap=secondary, sw=sw, phase=phase)
        else:
            self._set_state(ind=secondary, cap=primary, sw=sw, phase=phase)

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
            swr = self.swr
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
            swr = self.swr
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
        center = best_s

        while True:
            improved = False
            for sign in (-1, 1):
                s_candidate = center + sign * offset
                if s_candidate < 0 or s_candidate > 127:
                    continue

                # Start from best primary found so far
                self._bg_apply_state(sw, best_p, s_candidate, "bg_sec_start")
                start_swr = self.swr
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
                            swr_try = self.swr
                            if swr_try < 900:
                                if best_trial is None or swr_try < best_trial[1]:
                                    best_trial = (p, swr_try)
                                found = True
                        if found and best_trial is not None:
                            p_start = best_trial[0]
                            self._bg_apply_state(sw, p_start, s_candidate, "bg_sec_seed")
                            break

                p_best_local, swr_local = self._bg_walk_primary(
                    sw, p_start, s_candidate, self.swr, "bg_sec_walk"
                )
                # Apply/log best for this secondary
                self._bg_apply_state(sw, p_best_local, s_candidate, "bg_sec_best")
                if swr_local < global_best:
                    global_best = swr_local
                    best_p = p_best_local
                    best_s = s_candidate
                    improved = True
                    center = best_s

            if not improved:
                break
            offset = 1  # reset to immediate neighbors around new center

        self._bg_apply_state(sw, best_p, best_s, "bg_final_candidate")
        return best_p, best_s, global_best

    def _tune_bg(self) -> None:
        # Clear trace if needed
        if self.options.trace_steps:
            self.trace.clear()
            self._trace_step = 0

        self._update_measurement()
        self._trace("bg_start")

        ind_candidates = [0, 16, 32, 48, 64, 80, 96, 112, 127]
        cap_candidates = [0, 1, 2, 3, 5, 7, 9, 11, 15, 19, 23, 27, 43, 59, 75, 91, 123]

        best_primary_sw0 = best_secondary_sw0 = None
        best_primary_sw1 = best_secondary_sw1 = None
        best_swr_sw0 = best_swr_sw1 = 999

        for sw in (0, 1):
            for ind in ind_candidates:
                for cap in cap_candidates:
                    self._bg_apply_state(sw, ind if sw == 0 else cap, cap if sw == 0 else ind, "bg_coarse_eval")
                    swr = self.swr
                    if sw == 0:
                        if swr < best_swr_sw0:
                            best_swr_sw0 = swr
                            best_primary_sw0 = ind
                            best_secondary_sw0 = cap
                    else:
                        if swr < best_swr_sw1:
                            best_swr_sw1 = swr
                            best_primary_sw1 = cap
                            best_secondary_sw1 = ind

        coarse_results = []
        # Log coarse best per topology
        for sw_val, p_val, s_val, swr_val in (
            (0, best_primary_sw0, best_secondary_sw0, best_swr_sw0),
            (1, best_primary_sw1, best_secondary_sw1, best_swr_sw1),
        ):
            if p_val is not None and s_val is not None:
                self._bg_apply_state(sw_val, p_val, s_val, "bg_coarse_best")
                coarse_results.append((sw_val, p_val, s_val, swr_val))

        # Refine for any sw with coarse < 999
        refined_results = []
        for sw_val, p_val, s_val, swr_val in coarse_results:
            if swr_val >= 999:
                continue
            self._bg_apply_state(sw_val, p_val, s_val, "bg_refine_start")
            p_best, swr_best = self._bg_walk_primary(
                sw_val, p_val, s_val, swr_val, "bg_primary"
            )
            p_best, s_best, swr_best = self._bg_search_secondary(
                sw_val, p_best, s_val, swr_best
            )
            refined_results.append((swr_best, sw_val, p_best, s_best))

        if refined_results:
            refined_results.sort(key=lambda x: x[0])
            best = refined_results[0]
            _, sw_final, p_final, s_final = best
            self._bg_apply_state(sw_final, p_final, s_final, "bg_final_best")
        elif coarse_results:
            # choose best coarse if no refinement possible
            coarse_results.sort(key=lambda x: x[3])
            sw_final, p_final, s_final, _ = coarse_results[0]
            self._bg_apply_state(sw_final, p_final, s_final, "bg_final_best")
        else:
            self.atu_reset()

        self._trace("bg_end")

    # ---- public entrypoints ----
    def run_atu10(self, freq_hz: float, z_load: complex) -> AlgoResult:
        self._prepare_run(freq_hz, z_load)
        self._tune_atu10()
        return self._result()

    def run_bg(self, freq_hz: float, z_load: complex) -> AlgoResult:
        self._prepare_run(freq_hz, z_load)
        self._tune_bg()
        return self._result()
