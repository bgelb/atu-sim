from __future__ import annotations

import unittest

from atu10_firmware_sim.core import SimFlags, TunerSim
from atu10_firmware_sim.edz_example import TABLE1, TABLE3


class TestVSWRRegression(unittest.TestCase):
    """Regression checks on final tuned SWR for Cebik tables."""

    EXPECTED = {
        "bg": {
            "table1": {
                "3.6 MHz": 112,
                "3.9 MHz": 162,
                "7.0 MHz": 112,
                "10.1 MHz": 117,
                "14.0 MHz": 119,
            },
            "table3": {
                "3.6 MHz (70 ft)": 108,
                "3.9 MHz (70 ft)": 180,
                "7.0 MHz (70 ft)": 109,
                "10.1 MHz (70 ft)": 107,
                "14.0 MHz (70 ft)": 108,
            },
        },
        "atu10": {
            "table1": {
                "3.6 MHz": 207,
                "3.9 MHz": 999,
                "7.0 MHz": 999,
                "10.1 MHz": 999,
                "14.0 MHz": 999,
            },
            "table3": {
                "3.6 MHz (70 ft)": 167,
                "3.9 MHz (70 ft)": 999,
                "7.0 MHz (70 ft)": 999,
                "10.1 MHz (70 ft)": 999,
                "14.0 MHz (70 ft)": 999,
            },
        },
    }

    def _run_table(self, table, algo: str) -> dict[str, int]:
        swrs: dict[str, int] = {}
        for label, freq, z_load in table:
            sim = TunerSim(freq_hz=freq, z_load=z_load, flags=SimFlags(algorithm=algo))
            sim.atu_reset()
            sim.tune()
            swrs[label] = sim.SWR
        return swrs

    def _assert_table(self, table_name: str, table, algo: str) -> None:
        expected = self.EXPECTED[algo][table_name]
        actual = self._run_table(table, algo)
        self.assertEqual(expected, actual)

    def test_bg_table1(self) -> None:
        self._assert_table("table1", TABLE1, "bg")

    def test_bg_table3(self) -> None:
        self._assert_table("table3", TABLE3, "bg")

    def test_atu10_table1(self) -> None:
        self._assert_table("table1", TABLE1, "atu10")

    def test_atu10_table3(self) -> None:
        self._assert_table("table3", TABLE3, "atu10")


if __name__ == "__main__":
    unittest.main()
