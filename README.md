# ATU-10 Firmware Simulator

This is a small Python project that models:

- The LC relay bank of an ATU-10-style L-network tuner
- The N7DDC-style tuning algorithm (as implemented in ATU-10 FW 1.6)
- The resulting SWR as seen by a 50 Ω radio

It also includes an example script (`edz_example.py`) that evaluates
the tuner against the feedpoint impedances of an 88' #12 copper
doublet from Cebik's tables (free-space and 70' over ground).
The intent is to see how the firmware behaves when asked to tune that
common doublet directly at the feedpoint (no intervening feedline),
comparing the ideal brute-force LC match against what the ATU-10
algorithm actually converges to across the published impedances.

> NOTE: This is a **simulation**. It ignores detector noise, relay
> timing, and many hardware details, but the algorithm structure
> closely follows the firmware C code.

## Quick start

```bash
# (optional) create a virtualenv
python -m venv .venv
source .venv/bin/activate

# install dependencies (matplotlib)
pip install -e .

python -m atu10_firmware_sim.edz_example
```

This prints both Cebik tables using firmware-faithful defaults, then reruns
the 7.0 MHz (70 ft) point with tracing enabled, and emits SWR grid PNGs for
**every** table entry (both SW topologies) into `./plots` by default. Any
coarse debug you enable will appear only in the detailed 7.0 MHz (70 ft) rerun.

Each row includes:

- The best SWR the LC bank could achieve (brute-force search)
- The SWR that the **actual firmware algorithm** converges to
- The final relay state (inductor and capacitor bitmasks, topology)
- SWR grid PNGs for both topologies with the tuning path overlaid

CLI toggles (see `-h` for the full list):

```bash
# Enable coarse debug prints and force all coarse strategies to run (SWR capped by default)
python -m atu10_firmware_sim.edz_example --debug-coarse --force-all-coarse-strategies

# Just list all available flags and defaults
python -m atu10_firmware_sim.edz_example --list-flags

# Render coarse debug and SWR heatmaps (PNGs go to ./plots unless overridden)
python -m atu10_firmware_sim.edz_example --debug-coarse

# Choose a different output directory for the PNGs
python -m atu10_firmware_sim.edz_example --output-dir /tmp/atu-plots

# Skip PNG generation entirely
python -m atu10_firmware_sim.edz_example --skip-plots

# Use the reference ATU-10 firmware algorithm instead of the BG search
python -m atu10_firmware_sim.edz_example --algorithm atu10

# Run regression tests (uses current default BG and reference ATU-10 algos)
pytest

## New architecture (in progress)

Core pieces now being factored for a general relay-switched L-network tuner:

- `LCBank`: parameterized inductor/capacitor relay bank with impedance/SWR helpers.
- `Detector`: converts input impedance to a metric (e.g., `ATU10IntegerVSWRDetector`).
- `TuningAlgo`: pluggable tuning strategies (legacy ATU-10, BG wrappers for now).
- `ATUSimulator`: orchestrates bank + detector + tuning algorithm and emits a trace.
- `Plot` helpers: reuse the simulator trace to overlay tuning paths on SWR grids.

Existing examples still run, and regression tests guard BG/ATU-10 final SWR.
```

## Simulation flags

`SimFlags` toggles optional behaviors that default to firmware-faithful
settings. Current knobs:

- `force_all_coarse_strategies`: Always run coarse strategies 2/3 even if the
  firmware would skip them.
- `debug_coarse`: Print coarse-tune step-by-step debugging from within the
  simulator (off by default).
- `trace_steps`: Record every relay-set step with SW/ind/cap/SWR (used by the
  example to overlay the tuning path on the SWR heatmaps).
- `algorithm`: `"bg"` (default improved search) or `"atu10"` (firmware-faithful).

## BG tuning algorithm (default)

1) Coarse grid: evaluate a sparse LC grid for both topologies (SW=0 and SW=1);
   pick the best coarse point per topology (<9.99 SWR) and keep both if valid.
2) Primary walk: starting from the coarse winner, walk the primary axis
   (L for SW=0, C for SW=1) step-by-step until SWR worsens by >0.2.
3) Secondary neighbors: walk immediate neighbors of the secondary axis
   (C for SW=0, L for SW=1) one-by-one in both directions, each time doing a
   primary walk to find the best for that neighbor. Expand outward until no
   neighbor improves the global best.
4) Choose the best refined result across SW=0/1. The plots show coarse best
   (gold triangle), per-secondary bests (pink diamonds), and the final result
   (red star), along with the path.

## SWR grids

`swr_grid()` produces a 128x128 SWR map for each topology (SW=0 and SW=1),
covering every inductor/capacitor bitmask combination. The example prints
these as PNG heatmaps for every table entry. Rows are inductor bitmasks,
columns are capacitor bitmasks; color buckets correspond to
<1.2, <1.3, <1.4, <1.5, <1.7, <2, <2.5, <3, <4, <5, <6.5, <8, <9.5, and ≥9.5
SWR (light gray). The tuning path (coarse vs. sharp steps) is overlaid on each plot.
