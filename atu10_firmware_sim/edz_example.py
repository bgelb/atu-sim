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
    if R <= 0:
        return None
    delta = R * (z0 - R)
    if delta <= 0:
        return None
    w = 2 * math.pi * freq_hz
    root = math.sqrt(delta)
    candidates = [-X + root, -X - root]

    for xs in candidates:
        if xs <= 0:
            continue
        X_tot = X + xs
        denom = (R * R) + (X_tot * X_tot)
        B = X_tot / denom
        if B <= 0:
            continue
        L = xs / w
        C = B / w
        return {"sw": 0, "L": L, "C": C}
    return None


def solve_shunt_then_series(
    freq_hz: float, z_load: complex, z0: float
) -> dict | None:
    R = z_load.real
    X = z_load.imag
    if R <= 0:
        return None
    w = 2 * math.pi * freq_hz
    denom_base = (R * R) + (X * X)
    G = R / denom_base
    B_load = -X / denom_base
    delta = (G / z0) - (G * G)
    if delta <= 0:
        return None
    root = math.sqrt(delta)
    candidates = [-B_load + root, -B_load - root]

    for Bc in candidates:
        if Bc < 0:
            continue
        B_total = B_load + Bc
        denom = (G / z0)
        Xs = B_total / denom
        if Xs <= 0:
            continue
        L = Xs / w
        C = Bc / w
        return {"sw": 1, "L": L, "C": C}
    return None


def find_ideal_match(freq_hz: float, z_load: complex, z0: float) -> dict | None:
    solutions = []
    s1 = solve_series_then_shunt(freq_hz, z_load, z0)
    if s1:
        solutions.append(s1)
    s2 = solve_shunt_then_series(freq_hz, z_load, z0)
    if s2:
        solutions.append(s2)

    if not solutions:
        return None

    return min(solutions, key=lambda s: s["L"] + s["C"])


def run_table(table, title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(
        f"{'Freq':>12} | {'Z_load (R+jX)':>22} | {'Best LC SWR':>10} | {'Algo SWR':>10}"
    )
    print("-" * 70)

    for label, freq, zL in table:
        sim = TunerSim(freq_hz=freq, z_load=zL)
        sim.atu_reset()

        best_swr, best_state = brute_force_best(freq, zL)
        sim.tune()

        sign = "+" if zL.imag >= 0 else "-"
        print(
            f"{label:>12} | "
            f"{zL.real:6.0f} {sign} j{abs(zL.imag):4.0f} | "
            f"{fmt_swr(best_swr):>10} | "
            f"{fmt_swr(sim.SWR):>10} "
            f"(ind={sim.ind:3d}, cap={sim.cap:3d}, SW={sim.SW})"
        )

        ideal = find_ideal_match(freq, zL, sim.z0)
        bank: LCBank = sim.bank

        if ideal:
            z_ideal = continuous_input_impedance(
                freq, zL, ideal["L"], ideal["C"], ideal["sw"]
            )
            swr_ideal = swr_from_z(z_ideal, sim.z0)
            l_bits, l_val, c_bits, c_val, in_range = bank.nearest_lc(
                ideal["L"], ideal["C"]
            )
            z_near = l_network_input_impedance(freq, zL, l_bits, c_bits, ideal["sw"])
            swr_near = swr_from_z(z_near, sim.z0)
            print(
                f"    Ideal match (SW={ideal['sw']}): "
                f"L={ideal['L']*1e6:.3f} uH, C={ideal['C']*1e12:.1f} pF, "
                f"Z_in={z_ideal.real:6.1f} + j{z_ideal.imag:6.1f}, "
                f"SWR={fmt_swr(swr_ideal)}"
            )
            print(
                f"    Nearest discrete: "
                f"L={l_val*1e6:.3f} uH (bits={l_bits:03d}), "
                f"C={c_val*1e12:.1f} pF (bits={c_bits:03d}), "
                f"in_range={in_range}, "
                f"Z_in={z_near.real:6.1f} + j{z_near.imag:6.1f}, "
                f"SWR={fmt_swr(swr_near)}"
            )
        else:
            print("    Ideal match: not solvable with simple L-network.")

        z_alg = l_network_input_impedance(freq, zL, sim.ind, sim.cap, sim.SW)
        print(
            f"    Algo choice: L={bank.l_from_bits(sim.ind)*1e6:.3f} uH, "
            f"C={bank.c_from_bits(sim.cap)*1e12:.1f} pF, "
            f"Z_in={z_alg.real:6.1f} + j{z_alg.imag:6.1f}, "
            f"SWR={fmt_swr(sim.SWR)}"
        )


def main() -> None:
    run_table(TABLE1, "Cebik Table 1 - Free-space 88' doublet")
    run_table(TABLE3, "Cebik Table 3 - 70 ft high 88' doublet")


if __name__ == "__main__":
    main()
