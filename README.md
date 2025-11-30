# ATU-10 Firmware Simulator

This is a Python project that simulates relay-switched L-network autotuners
(e.g., ATU-10) and their tuning algorithms. It includes:

- A parameterized LC relay bank model
- Pluggable detectors (e.g., ATU-10 integer SWR)
- Pluggable tuning algorithms (reference ATU-10 firmware, BG improved search)
- A simulator that orchestrates bank + detector + algorithm and emits traces
- An example that evaluates the Cebik 88' doublet tables and renders SWR grids

> NOTE: This is a **simulation**. It ignores detector noise, relay
> timing, and many hardware details, but the algorithm structure
> closely follows the firmware C code.

## Quick start

```bash
# (optional) create a virtualenv
python -m venv .venv
source .venv/bin/activate

# install dependencies (matplotlib + pytest)
pip install -e '.[dev]'

# run the example (prints tables + SWR heatmaps into ./plots)
python -m examples.edz_example
```

The example prints both Cebik tables using the default BG algorithm, then emits
SWR grid PNGs for every table entry (both SW topologies) into `./plots` by
default with the tuning path overlaid.

CLI toggles (see `-h` for the full list):

```bash
# Choose a different output directory for the PNGs
python -m examples.edz_example --output-dir /tmp/atu-plots

# Skip PNG generation entirely
python -m examples.edz_example --skip-plots

# Use the reference ATU-10 firmware algorithm instead of the BG search
python -m examples.edz_example --algorithm atu10

# Run regression tests (uses current default BG and reference ATU-10 algos)
pytest
```

## Architecture

Core pieces for a general relay-switched L-network tuner:

- `LCBank` (`atu_sim/lc_bank.py`): parameterized inductor/capacitor relay
  bank with impedance helpers (`atu10_bank()` builds the ATU-10 values).
- `Detector` (`detectors.py`): converts input impedance to a metric
  (e.g., `ATU10IntegerVSWRDetector` with the ATU-10 integer SWR quantizer).
- `TuningAlgo` implementations (`tuning_algos/bg_algo.py`, `tuning_algos/atu10_reference.py`):
  pluggable tuning strategies wrapping the firmware-like state machine.
- `ATUSimulator`: orchestrates bank + detector + tuning algorithm and emits a trace
  of each relay state with computed impedance/SWR/detector output.
- `examples/edz_example.py`: evaluates the simulator on Cebik tables and renders
  SWR heatmaps with the tuning path overlaid.

## SWR grids

`examples/edz_example.py` includes a helper that produces a 128x128 SWR map for
each topology (SW=0 and SW=1), covering every inductor/capacitor bitmask
combination. The example prints these as PNG heatmaps for every table entry.
Rows are inductor bitmasks, columns are capacitor bitmasks; color buckets
correspond to <1.2, <1.3, <1.4, <1.5, <1.7, <2, <2.5, <3, <4, <5, <6.5, <8,
<9.5, and ≥9.5 SWR (light gray). The tuning path (coarse vs. sharp steps) is
overlaid on each plot.
