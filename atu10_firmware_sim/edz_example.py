from __future__ import annotations

import math

from .core import LCBank, TunerSim, brute_force_best, l_network_input_impedance, swr_from_z


TABLE1 = [
    ("3.6 MHz", 3.6e6, complex(25, -615)),
    ("3.9 MHz", 3.9e6, complex(30, -500)),
    ("7.0 MHz", 7.0e6, complex(185, 510)),
    ("10.1 MHz", 10.1e6, complex(3360, 2245)),
    ("14.0 MHz", 14.0e6, complex(155, -805)),
]

TABLE3 = [
    ("3.6 MHz (70 ft)", 3.6e6, complex(30, -610)),
    ("3.9 MHz (70 ft)", 3.9e6, complex(35, -495)),
    ("7.0 MHz (70 ft)", 7.0e6, complex(165, 485)),
    ("10.1 MHz (70 ft)", 10.1e6, complex(3810, 2160)),
    ("14.0 MHz (70 ft)", 14.0e6, complex(155, -820)),
]


def fmt_swr(swr_int: int) -> str:
    if swr_int >= 999:
        return ">= 9.99"
    return f"{swr_int / 100.0:.2f}"


def continuous_input_impedance(
    freq_hz: float, z_load: complex, L: float, C: float, sw: int
) -> complex:
    w = 2 * math.pi * freq_hz
    j = 1j

    z_L_series = j * w * L if L > 0 else 0j
    y_C = j * w * C if C > 0 else 0j

    if sw == 0:
        y_total = (1.0 / z_load) + y_C
        if abs(y_total) < 1e-18:
            z_node = complex(1e12, 0)
        else:
            z_node = 1.0 / y_total
        return z_L_series + z_node

    z_series = z_L_series + z_load
    y_series = 1.0 / z_series if abs(z_series) > 1e-18 else 0j
    y_in = y_series + y_C
    if abs(y_in) < 1e-18:
        return complex(1e12, 0)
    return 1.0 / y_in


def solve_series_then_shunt(
    freq_hz: float, z_load: complex, z0: float
) -> dict | None:
    R = z_load.real
    X = z_load.imag
    if R <= 0 or R > z0:
        return None
    w = 2 * math.pi * freq_hz
    term = (R * (z0 - R))
    if term < 0:
        return None
    Xs = math.sqrt(term) - X  # choose branch that yields Bc > 0
    X_tot = X + Xs
    denom = (R * R) + (X_tot * X_tot)
    G = R / denom
    Bc = (X + Xs) / denom
    if Bc <= 0 or abs(G - (1.0 / z0)) > 1e-9:
        return None
    L = Xs / w
    C = Bc / w
    return {"sw": 0, "L": L, "C": C}


def solve_shunt_then_series(
    freq_hz: float, z_load: complex, z0: float
) -> dict | None:
    R = z_load.real
    X = z_load.imag
    if R <= 0 or R < z0:
        return None
    w = 2 * math.pi * freq_hz
    denom_base = (R * R) + (X * X)
    G = R / denom_base
    B_load = -X / denom_base
    delta = (G / z0) - (G * G)
    if delta < 0:
        return None
    B_target = math.sqrt(delta)
    Bc = B_target - B_load
    if Bc < 0:
        return None
    Xs = B_target * z0 / G
    if Xs <= 0:
        return None
    L = Xs / w
    C = Bc / w
    return {"sw": 1, "L": L, "C": C}


def find_ideal_match(freq_hz: float, z_load: complex, z0: float) -> dict:
    """
    Analytic L-network match (both topologies), unconstrained by relay steps.
    """
    R = z_load.real
    X = z_load.imag
    w = 2 * math.pi * freq_hz
    candidates: list[dict] = []

    # sw=0 in simulator: shunt first (at load), then series
    denom_base = (R * R) + (X * X)
    if denom_base > 0:
        G = R / denom_base
        B_load = -X / denom_base
        if 0 < G < (1.0 / z0):
            target = (G - z0 * G * G) / z0
            if target >= 0:
                root = math.sqrt(target)
                for sign in (1.0, -1.0):
                    B_total = sign * root
                    Bc = B_total - B_load
                    if Bc == 0:
                        continue
                    Xs = B_total / (G * G + B_total * B_total)
                    L_series = Xs / w
                    C_shunt = Bc / w  # negative => shunt L
                    z_in = continuous_input_impedance(freq_hz, z_load, L_series, C_shunt, 0)
                    swr = swr_from_z(z_in, z0)
                    candidates.append(
                        {"sw": 0, "L": L_series, "C": C_shunt, "z_in": z_in, "swr": swr}
                    )

    # sw=1 in simulator: series first, then shunt at input
    if R > 0 and R < z0:
        term = (R * z0) - (R * R)
        if term >= 0:
            root = math.sqrt(term)
            for sign in (1.0, -1.0):
                Xs = -X + sign * root
                X_tot = X + Xs
                denom = (R * R) + (X_tot * X_tot)
                if denom == 0:
                    continue
                Bc = X_tot / denom  # shunt susceptance
                if Bc == 0:
                    continue
                L_series = Xs / w
                C_shunt = Bc / w
                z_in = continuous_input_impedance(freq_hz, z_load, L_series, C_shunt, 1)
                swr = swr_from_z(z_in, z0)
                candidates.append(
                    {"sw": 1, "L": L_series, "C": C_shunt, "z_in": z_in, "swr": swr}
                )

    if not candidates:
        z_in = z_load
        return {"sw": 0, "L": 0.0, "C": 0.0, "z_in": z_in, "swr": swr_from_z(z_in, z0)}

    return min(candidates, key=lambda c: c["swr"])


def run_table(table, title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    freq_width = max(len("Freq"), max(len(label) for label, _, _ in table))
    load_width = 18
    ideal_width = 88
    best_width = 88
    algo_width = 88
    print(
        f"{'Freq':>{freq_width}} | {'Z_load (R+jX)':>{load_width}} | "
        f"{'Ideal (L/C, topology, Zin, SWR)':<{ideal_width}}| "
        f"{'Best discrete (L/C, topology, Zin, SWR)':<{best_width}}| "
        f"{'Tune algo (L/C, topology, Zin, SWR)':<{algo_width}}"
    )
    print("-" * (freq_width + 3 + load_width + 3 + ideal_width + 2 + best_width + 2 + algo_width))

    for label, freq, zL in table:
        sim = TunerSim(freq_hz=freq, z_load=zL)
        sim.atu_reset()

        verbose_coarse = "7.0 MHz (70 ft)" in label

        best_swr, best_state = brute_force_best(freq, zL)
        sim.tune()
        bank: LCBank = sim.bank

        def topo_label(sw: int, C_val: float) -> str:
            shunt = "shunt-C" if C_val > 0 else ("open" if C_val == 0 else "shunt-L")
            return f"{shunt}@{'load' if sw == 0 else 'input'}"

        ideal = find_ideal_match(freq, zL, sim.z0)
        z_ideal = ideal["z_in"]
        ideal_str = (
            f"L={ideal['L']*1e6:8.3f}u "
            f"C={ideal['C']*1e12:8.1f}p "
            f"{topo_label(ideal['sw'], ideal['C']):>12} "
            f"Zin={z_ideal.real:9.2f}+j{z_ideal.imag:9.2f} "
            f"SWR={fmt_swr(ideal['swr']):>6}"
        )

        z_best = l_network_input_impedance(
            freq, zL, best_state[0], best_state[1], best_state[2], bank
        )
        best_str = (
            f"L={bank.l_from_bits(best_state[0])*1e6:8.3f}u "
            f"C={bank.c_from_bits(best_state[1])*1e12:8.1f}p "
            f"{topo_label(best_state[2], bank.c_from_bits(best_state[1])):>12} "
            f"Zin={z_best.real:9.2f}+j{z_best.imag:9.2f} "
            f"SWR={fmt_swr(best_swr):>6}"
        )

        z_alg = l_network_input_impedance(freq, zL, sim.ind, sim.cap, sim.SW)
        algo_str = (
            f"L={bank.l_from_bits(sim.ind)*1e6:8.3f}u "
            f"C={bank.c_from_bits(sim.cap)*1e12:8.1f}p "
            f"{topo_label(sim.SW, bank.c_from_bits(sim.cap)):>12} "
            f"Zin={z_alg.real:9.2f}+j{z_alg.imag:9.2f} "
            f"SWR={fmt_swr(sim.SWR):>6}"
        )

        sign = "+" if zL.imag >= 0 else "-"
        load_str = f"{zL.real:7.0f} {sign} j{abs(zL.imag):5.0f}"
        print(
            f"{label:>{freq_width}} | "
            f"{load_str:>{load_width}} | "
            f"{ideal_str:<{ideal_width}}| "
            f"{best_str:<{best_width}}| "
            f"{algo_str:<{algo_width}}"
        )

        if verbose_coarse:
            def coarse_debug(sw_val: int) -> None:
                print(f"    Coarse sweep debug (reset, {'shunt@load' if sw_val==0 else 'shunt@input'}):")
                sim_dbg = TunerSim(freq_hz=freq, z_load=zL)
                sim_dbg.atu_reset()
                sim_dbg.SW = sw_val
                sim_dbg.relay_set()
                print(f"      start: SWR={fmt_swr(sim_dbg.SWR)} ind={sim_dbg.ind:3d} cap={sim_dbg.cap:3d}")

                def strat_label(n: int) -> None:
                    print(f"      Strategy {n}:")

                # Strategy 1: coarse_cap then coarse_ind
                strat_label(1)
                sim_dbg.cap = 0
                sim_dbg.ind = 0
                sim_dbg.relay_set()
                # coarse_cap
                sim_dbg.get_swr()
                swr_mem = sim_dbg.SWR // 10
                cap_mem = 0
                cap = 1
                while cap < 64:
                    sim_dbg.relay_set(cap=cap)
                    swr_scaled = sim_dbg.SWR // 10
                    print(f"        cap step: cap={cap:3d} SWR={fmt_swr(sim_dbg.SWR)} scaled={swr_scaled}")
                    if swr_scaled <= swr_mem:
                        cap_mem = cap
                        swr_mem = swr_scaled
                        cap *= 2
                    else:
                        break
                sim_dbg.cap = cap_mem
                sim_dbg.relay_set()
                print(f"        coarse_cap chosen cap={sim_dbg.cap:3d} SWR={fmt_swr(sim_dbg.SWR)}")

                # coarse_ind
                sim_dbg.get_swr()
                swr_mem = sim_dbg.SWR // 10
                ind_mem = 0
                ind = 1
                while ind < 64:
                    sim_dbg.relay_set(ind=ind)
                    swr_scaled = sim_dbg.SWR // 10
                    print(f"        ind step: ind={ind:3d} SWR={fmt_swr(sim_dbg.SWR)} scaled={swr_scaled}")
                    if swr_scaled <= swr_mem:
                        ind_mem = ind
                        swr_mem = swr_scaled
                        ind *= 2
                    else:
                        break
                sim_dbg.ind = ind_mem
                sim_dbg.relay_set()
                print(f"        coarse_ind chosen ind={sim_dbg.ind:3d} SWR={fmt_swr(sim_dbg.SWR)}")

                # Strategy 2: coarse_ind then coarse_cap (only if cap<=2 and ind<=2)
                if sim_dbg.cap <= 2 and sim_dbg.ind <= 2:
                    strat_label(2)
                    sim_dbg.cap = 0
                    sim_dbg.ind = 0
                    sim_dbg.relay_set()
                    # coarse_ind
                    sim_dbg.get_swr()
                    swr_mem = sim_dbg.SWR // 10
                    ind_mem = 0
                    ind = 1
                    while ind < 64:
                        sim_dbg.relay_set(ind=ind)
                        swr_scaled = sim_dbg.SWR // 10
                        print(f"        ind step: ind={ind:3d} SWR={fmt_swr(sim_dbg.SWR)} scaled={swr_scaled}")
                        if swr_scaled <= swr_mem:
                            ind_mem = ind
                            swr_mem = swr_scaled
                            ind *= 2
                        else:
                            break
                    sim_dbg.ind = ind_mem
                    sim_dbg.relay_set()
                    print(f"        coarse_ind chosen ind={sim_dbg.ind:3d} SWR={fmt_swr(sim_dbg.SWR)}")
                    # coarse_cap
                    sim_dbg.get_swr()
                    swr_mem = sim_dbg.SWR // 10
                    cap_mem = 0
                    cap = 1
                    while cap < 64:
                        sim_dbg.relay_set(cap=cap)
                        swr_scaled = sim_dbg.SWR // 10
                        print(f"        cap step: cap={cap:3d} SWR={fmt_swr(sim_dbg.SWR)} scaled={swr_scaled}")
                        if swr_scaled <= swr_mem:
                            cap_mem = cap
                            swr_mem = swr_scaled
                            cap *= 2
                        else:
                            break
                    sim_dbg.cap = cap_mem
                    sim_dbg.relay_set()
                    print(f"        coarse_cap chosen cap={sim_dbg.cap:3d} SWR={fmt_swr(sim_dbg.SWR)}")
                else:
                    print("      Strategy 2 skipped (cap>2 or ind>2 after Strategy 1)")

                # Strategy 3: coarse_ind_cap (only if cap<=2 and ind<=2)
                if sim_dbg.cap <= 2 and sim_dbg.ind <= 2:
                    strat_label(3)
                    sim_dbg.cap = 0
                    sim_dbg.ind = 0
                    sim_dbg.relay_set()
                    sim_dbg.get_swr()
                    swr_mem = sim_dbg.SWR // 10
                    ind_mem = 0
                    ind = 1
                    while ind < 64:
                        sim_dbg.relay_set(ind=ind, cap=ind)
                        swr_scaled = sim_dbg.SWR // 10
                        print(f"        ind=cap step: val={ind:3d} SWR={fmt_swr(sim_dbg.SWR)} scaled={swr_scaled}")
                        if swr_scaled <= swr_mem:
                            ind_mem = ind
                            swr_mem = swr_scaled
                            ind *= 2
                        else:
                            break
                    sim_dbg.ind = ind_mem
                    sim_dbg.cap = ind_mem
                    sim_dbg.relay_set()
                    print(f"        coarse_ind_cap chosen ind=cap={sim_dbg.ind:3d} SWR={fmt_swr(sim_dbg.SWR)}")
                else:
                    print("      Strategy 3 skipped (cap>2 or ind>2 after Strategy 1)")

            coarse_debug(0)
            coarse_debug(1)


def main() -> None:
    run_table(TABLE1, "Cebik Table 1 - Free-space 88' doublet")
    run_table(TABLE3, "Cebik Table 3 - 70 ft high 88' doublet")


if __name__ == "__main__":
    main()
