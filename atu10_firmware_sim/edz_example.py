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

    # Topology sw=0: series L then shunt C
    if R > 0 and R <= z0:
        term = (R * z0) - (R * R)
        if term >= 0:
            for sign in (1.0, -1.0):
                Xs = -X + sign * math.sqrt(term)
                X_tot = X + Xs
                denom = (R * R) + (X_tot * X_tot)
                if denom == 0:
                    continue
                Bc = X_tot / denom
                if Bc <= 0:
                    continue
                L = Xs / w
                C = Bc / w
                if L < 0 or C < 0:
                    continue
                z_in = continuous_input_impedance(freq_hz, z_load, L, C, 0)
                swr = swr_from_z(z_in, z0)
                candidates.append({"sw": 0, "L": L, "C": C, "z_in": z_in, "swr": swr})

    # Topology sw=1: shunt C then series L
    denom_base = (R * R) + (X * X)
    if denom_base > 0:
        G = R / denom_base
        B_load = -X / denom_base
        if 0 < G < (1.0 / z0):
            numerator = G - (z0 * G * G)
            if numerator >= 0:
                Btot = math.sqrt(numerator / z0)
                for sign in (1.0, -1.0):
                    Bc = sign * Btot - B_load
                    if Bc <= 0:
                        continue
                    y_total = complex(G, B_load + Bc)
                    if y_total == 0:
                        continue
                    z_t = 1.0 / y_total
                    Xs = -z_t.imag
                    L = Xs / w
                    C = Bc / w
                    if L < 0 or C < 0:
                        continue
                    z_in = z_t + 1j * Xs
                    swr = swr_from_z(z_in, z0)
                    candidates.append(
                        {"sw": 1, "L": L, "C": C, "z_in": z_in, "swr": swr}
                    )

    if not candidates:
        # Fallback: return high SWR but with zero components
        z_in = z_load
        return {"sw": 0, "L": 0.0, "C": 0.0, "z_in": z_in, "swr": swr_from_z(z_in, z0)}

    return min(candidates, key=lambda c: c["swr"])


def run_table(table, title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    ideal_width = 64
    best_width = 64
    algo_width = 64
    print(
        f"{'Freq':>14} | {'Z_load (R+jX)':>22} | "
        f"{'Ideal (L/C, topology, Zin, SWR)':<{ideal_width}}| "
        f"{'Best discrete (L/C, topology, Zin, SWR)':<{best_width}}| "
        f"{'Tune algo (L/C, topology, Zin, SWR)':<{algo_width}}"
    )
    print("-" * (14 + 3 + 22 + 3 + ideal_width + 2 + best_width + 2 + algo_width))

    for label, freq, zL in table:
        sim = TunerSim(freq_hz=freq, z_load=zL)
        sim.atu_reset()

        best_swr, best_state = brute_force_best(freq, zL)
        sim.tune()
        bank: LCBank = sim.bank

        ideal = find_ideal_match(freq, zL, sim.z0)
        z_ideal = ideal["z_in"]
        ideal_str = (
            f"L={ideal['L']*1e6:7.3f}u "
            f"C={ideal['C']*1e12:8.1f}p "
            f"SW={ideal['sw']} "
            f"Zin={z_ideal.real:8.2f}+j{z_ideal.imag:8.2f} "
            f"SWR={fmt_swr(ideal['swr']):>6}"
        )

        sign = "+" if zL.imag >= 0 else "-"
        z_best = l_network_input_impedance(
            freq, zL, best_state[0], best_state[1], best_state[2], bank
        )
        best_str = (
            f"L={bank.l_from_bits(best_state[0])*1e6:7.3f}u "
            f"C={bank.c_from_bits(best_state[1])*1e12:8.1f}p "
            f"SW={best_state[2]} "
            f"Zin={z_best.real:8.2f}+j{z_best.imag:8.2f} "
            f"SWR={fmt_swr(best_swr):>6}"
        )

        z_alg = l_network_input_impedance(freq, zL, sim.ind, sim.cap, sim.SW)
        algo_str = (
            f"L={bank.l_from_bits(sim.ind)*1e6:7.3f}u "
            f"C={bank.c_from_bits(sim.cap)*1e12:8.1f}p "
            f"SW={sim.SW} "
            f"Zin={z_alg.real:8.2f}+j{z_alg.imag:8.2f} "
            f"SWR={fmt_swr(sim.SWR):>6}"
        )

        print(
            f"{label:>14} | "
            f"{zL.real:6.0f} {sign} j{abs(zL.imag):4.0f} | "
            f"{ideal_str:<{ideal_width}}| "
            f"{best_str:<{best_width}}| "
            f"{algo_str:<{algo_width}}"
        )


def main() -> None:
    run_table(TABLE1, "Cebik Table 1 - Free-space 88' doublet")
    run_table(TABLE3, "Cebik Table 3 - 70 ft high 88' doublet")


if __name__ == "__main__":
    main()
