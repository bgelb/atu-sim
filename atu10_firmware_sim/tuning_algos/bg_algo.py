from __future__ import annotations

from dataclasses import dataclass

from ..detectors import ATU10IntegerVSWRDetector, Detector
from ..lc_bank import LCBank, ShuntPosition
from .types import AlgoResult, AlgoTrace, Topology, TuningAlgo, TuningConfig, TuningPhase


def _topology_from_sw(sw: int) -> Topology:
    return Topology.SHUNT_AT_LOAD if sw == 0 else Topology.SHUNT_AT_SOURCE


@dataclass
class BGOptions:
    trace_steps: bool = True


class BGAlgo(TuningAlgo):
    """Improved BG search algorithm implemented directly."""

    def __init__(
        self, bank: LCBank, detector: Detector, options: BGOptions | None = None
    ) -> None:
        if len(bank.l_values) != 7 or len(bank.c_values) != 7:
            raise AssertionError("BGAlgo expects 7 L relays and 7 C relays")
        if not isinstance(detector, ATU10IntegerVSWRDetector):
            raise TypeError("BGAlgo requires ATU10IntegerVSWRDetector")
        self.bank = bank
        self.detector = detector
        self.options = options or BGOptions()
        self.trace: list[AlgoTrace] = []
        self._trace_step = 0
        self.freq_hz = 0.0
        self.z_load: complex = 0j
        self.ind = 0
        self.cap = 0
        self.sw = 0
        self.swr = 999
        self.z_in = complex(0, 0)

    # ---- helpers ----
    def _update_measurement(self) -> None:
        pos = ShuntPosition.LOAD if self.sw == 0 else ShuntPosition.SOURCE
        self.z_in = self.bank.input_impedance(
            self.freq_hz, self.z_load, self.ind, self.cap, pos
        )
        self.swr = self.detector.measure(self.z_in)

    def _trace(self, tuning_phase: TuningPhase) -> None:
        if not self.options.trace_steps:
            return
        self._trace_step += 1
        self.trace.append(
            AlgoTrace(
                step=self._trace_step,
                tuning_phase=tuning_phase,
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
        tuning_phase: TuningPhase | None = None,
    ) -> None:
        if ind is not None:
            self.ind = ind
        if cap is not None:
            self.cap = cap
        if sw is not None:
            self.sw = sw
        self._update_measurement()
        if tuning_phase:
            self._trace(tuning_phase)

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

    def atu_reset(self) -> None:
        self.ind = 0
        self.cap = 1
        self.sw = 0
        self._update_measurement()
        self._trace(TuningPhase.RESET)

    # ---- BG-specific helpers ----
    def _bg_apply_state(self, sw: int, primary: int, secondary: int, tuning_phase: TuningPhase) -> None:
        """
        Helper to set relays with a primary/secondary axis abstraction.
        For sw=0: primary=L(ind), secondary=C(cap).
        For sw=1: primary=C(cap), secondary=L(ind).
        """
        if sw == 0:
            self._set_state(ind=primary, cap=secondary, sw=sw, tuning_phase=tuning_phase)
        else:
            self._set_state(ind=secondary, cap=primary, sw=sw, tuning_phase=tuning_phase)

    def _bg_walk_primary(
        self, sw: int, primary_start: int, secondary: int, best_swr: int
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
            self._bg_apply_state(sw, p, secondary, TuningPhase.FINE_STEP)
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
            self._bg_apply_state(sw, p, secondary, TuningPhase.FINE_STEP)
            swr = self.swr
            if swr < best_val:
                best_val = swr
                best_primary = p
            elif swr > best_val + 20:
                break

        self._bg_apply_state(sw, best_primary, secondary, TuningPhase.FINE_BEST)
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
                self._bg_apply_state(sw, best_p, s_candidate, TuningPhase.FINE_START)
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
                            self._bg_apply_state(sw, p, s_candidate, TuningPhase.FINE_STEP)
                            swr_try = self.swr
                            if swr_try < 900:
                                if best_trial is None or swr_try < best_trial[1]:
                                    best_trial = (p, swr_try)
                                found = True
                        if found and best_trial is not None:
                            p_start = best_trial[0]
                            self._bg_apply_state(sw, p_start, s_candidate, TuningPhase.FINE_STEP)
                            break

                p_best_local, swr_local = self._bg_walk_primary(
                    sw, p_start, s_candidate, self.swr
                )
                # Apply/log best for this secondary
                self._bg_apply_state(sw, p_best_local, s_candidate, TuningPhase.SECONDARY_BEST)
                if swr_local < global_best:
                    global_best = swr_local
                    best_p = p_best_local
                    best_s = s_candidate
                    improved = True
                    center = best_s

            if not improved:
                break
            offset = 1  # reset to immediate neighbors around new center

        self._bg_apply_state(sw, best_p, best_s, TuningPhase.FINAL_CANDIDATE)
        return best_p, best_s, global_best

    def _tune_bg(self) -> None:
        # Clear trace if needed
        if self.options.trace_steps:
            self.trace.clear()
            self._trace_step = 0

        self._update_measurement()
        self._trace(TuningPhase.START)

        ind_candidates = [0, 16, 32, 48, 64, 80, 96, 112, 127]
        cap_candidates = [0, 1, 2, 3, 5, 7, 9, 11, 15, 19, 23, 27, 43, 59, 75, 91, 123]

        best_primary_sw0 = best_secondary_sw0 = None
        best_primary_sw1 = best_secondary_sw1 = None
        best_swr_sw0 = best_swr_sw1 = 999

        for sw in (0, 1):
            for ind in ind_candidates:
                for cap in cap_candidates:
                    self._bg_apply_state(sw, ind if sw == 0 else cap, cap if sw == 0 else ind, TuningPhase.COARSE_STEP)
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
                self._bg_apply_state(sw_val, p_val, s_val, TuningPhase.COARSE_BEST)
                coarse_results.append((sw_val, p_val, s_val, swr_val))

        # Refine for any sw with coarse < 999
        refined_results = []
        for sw_val, p_val, s_val, swr_val in coarse_results:
            if swr_val >= 999:
                continue
            self._bg_apply_state(sw_val, p_val, s_val, TuningPhase.FINE_START)
            p_best, swr_best = self._bg_walk_primary(
                sw_val, p_val, s_val, swr_val
            )
            p_best, s_best, swr_best = self._bg_search_secondary(
                sw_val, p_best, s_val, swr_best
            )
            refined_results.append((swr_best, sw_val, p_best, s_best))

        if refined_results:
            refined_results.sort(key=lambda x: x[0])
            best = refined_results[0]
            _, sw_final, p_final, s_final = best
            self._bg_apply_state(sw_final, p_final, s_final, TuningPhase.FINAL_BEST)
        elif coarse_results:
            # choose best coarse if no refinement possible
            coarse_results.sort(key=lambda x: x[3])
            sw_final, p_final, s_final, _ = coarse_results[0]
            self._bg_apply_state(sw_final, p_final, s_final, TuningPhase.FINAL_BEST)
        else:
            self.atu_reset()

        self._trace(TuningPhase.FINAL)

    # ---- public entrypoint ----
    def run(self, freq_hz: float, z_load: complex) -> AlgoResult:
        self._prepare_run(freq_hz, z_load)
        self._tune_bg()
        return self._result()

    def name(self) -> str:
        return "bg"
