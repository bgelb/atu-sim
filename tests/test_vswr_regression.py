from __future__ import annotations

import pytest

from examples.edz_example.cebik_tables import TABLE1, TABLE3
from atu_sim.hardware import atu10_bank_alt
from atu_sim.detectors import ATU10IntegerVSWRDetector
from atu_sim.simulator import ATUSimulator
from atu_sim.tuning_algos.atu10_reference import ATU10ReferenceAlgo
from atu_sim.tuning_algos.bg_algo import BGAlgo

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
    bank = atu10_bank_alt()
    detector = ATU10IntegerVSWRDetector()
    AlgoCls = BGAlgo if algo == "bg" else ATU10ReferenceAlgo
    for label, freq, z_load in table:
        algo_instance = AlgoCls(bank, detector)
        sim = ATUSimulator(algorithm=algo_instance)
        result = sim.tune(freq, z_load)
        swrs[label] = result.final_swr
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
