from __future__ import annotations

from .core import TunerSim, brute_force_best


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


def run_table(table, title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"{'Freq':>12} | {'Z_load (R+jX)':>22} | {'Best LC SWR':>10} | {'Algo SWR':>10}")
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


def main() -> None:
    run_table(TABLE1, "Cebik Table 1 - Free-space 88' doublet")
    run_table(TABLE3, "Cebik Table 3 - 70 ft high 88' doublet")


if __name__ == "__main__":
    main()
