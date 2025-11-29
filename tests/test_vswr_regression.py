from __future__ import annotations

import pytest

from atu10_firmware_sim.core import SimFlags, TunerSim
from atu10_firmware_sim.edz_example import TABLE1, TABLE3

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


def run_table(table, algo: str) -> dict[str, int]:
    swrs: dict[str, int] = {}
    for label, freq, z_load in table:
        sim = TunerSim(freq_hz=freq, z_load=z_load, flags=SimFlags(algorithm=algo))
        sim.atu_reset()
        sim.tune()
        swrs[label] = sim.SWR
    return swrs


@pytest.mark.parametrize(
    "algo,table_name,table",
    [
        ("bg", "table1", TABLE1),
        ("bg", "table3", TABLE3),
        ("atu10", "table1", TABLE1),
        ("atu10", "table3", TABLE3),
    ],
)
def test_vswr_regression(algo: str, table_name: str, table) -> None:
    expected = EXPECTED[algo][table_name]
    actual = run_table(table, algo)
    assert actual == expected
