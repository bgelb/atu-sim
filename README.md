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

## SWR grids

`swr_grid()` produces a 128x128 SWR map for each topology (SW=0 and SW=1),
covering every inductor/capacitor bitmask combination. The example prints
these as PNG heatmaps for every table entry. Rows are inductor bitmasks,
columns are capacitor bitmasks; color buckets correspond to
<1.2, <1.3, <1.4, <1.5, <1.7, <2, <2.5, <3, <4, <5, <6.5, <8, <9.5, and ≥9.5
SWR (light gray). The tuning path (coarse vs. sharp steps) is overlaid on each plot.
