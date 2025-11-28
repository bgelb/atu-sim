# ATU-10 Firmware Simulator

This is a small Python project that models:

- The LC relay bank of an ATU-10-style L-network tuner
- The N7DDC-style tuning algorithm (as implemented in ATU-10 FW 1.6)
- The resulting SWR as seen by a 50 Ω radio

It also includes an example script (`edz_example.py`) that evaluates
the tuner against the feedpoint impedances of an 88' #12 copper
doublet from Cebik's tables (free-space and 70' over ground).

> NOTE: This is a **simulation**. It ignores detector noise, relay
> timing, and many hardware details, but the algorithm structure
> closely follows the firmware C code.

## Quick start

```bash
python -m atu10_firmware_sim.edz_example
```

This will print, for each Cebik table entry:

- The best SWR the LC bank could achieve (brute-force search)
- The SWR that the **actual firmware algorithm** converges to
- The final relay state (inductor and capacitor bitmasks, topology)
