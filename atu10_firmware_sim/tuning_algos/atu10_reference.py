from __future__ import annotations

from dataclasses import dataclass

from ..detectors import Detector
from ..lc_bank import LCBank, ShuntPosition
from .types import AlgoResult, AlgoTrace, Topology, TuningAlgo, TuningConfig


def _topology_from_sw(sw: int) -> Topology:
    return Topology.SHUNT_AT_LOAD if sw == 0 else Topology.SHUNT_AT_SOURCE


@dataclass
class ATU10Options:
    force_all_coarse_strategies: bool = False
    debug_coarse: bool = False
    trace_steps: bool = True


class ATU10ReferenceAlgo(TuningAlgo):
    """Firmware-faithful ATU-10 algorithm implemented directly."""

    def __init__(
        self, bank: LCBank, detector: Detector, options: ATU10Options | None = None
    ) -> None:
        if len(bank.l_values) != 7 or len(bank.c_values) != 7:
            raise AssertionError("ATU10ReferenceAlgo expects 7 L relays and 7 C relays")
        self.bank = bank
        self.detector = detector
        self.options = options or ATU10Options()
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

    def atu_reset(self) -> None:
        self.ind = 0
        self.cap = 1
        self.sw = 0
        self._update_measurement()
        self._trace("reset")

    # ---- ATU-10 helpers ----
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

    # ---- public entrypoint ----
    def run(self, freq_hz: float, z_load: complex) -> AlgoResult:
        self._prepare_run(freq_hz, z_load)
        self._tune_atu10()
        return self._result()

    def name(self) -> str:
        return "atu10"
