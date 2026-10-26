# GeForce FP32-accumulate cap, measured on stock cards

Measured 2026-10-26 on RunPod, one pod per card, no unlock, no mod. Raw files in `results/`.

## Question

GeForce Ampere and Ada parts are said to run FP16/BF16 tensor math with FP32 accumulation at half
the rate of FP16 accumulation (NVIDIA documents this in the Ada whitepaper; open-source kernel
modules published 2026-10-25 lift it). Does a stock card show the cap?

**Hypothesis, written before the runs:** on both cards, FP16-acc32 / FP16-acc16 throughput is
about 0.5. Holds if at most 0.6 on large square shapes; fails if 0.8 or more.

## Method

- `run.sh`: cuBLAS `cublasGemmEx`, same card, five modes (FP16 acc16, FP16 acc32, BF16 acc32,
  TF32, INT8 acc32). Six shapes: three square, three Llama-3.1-8B prefill GEMMs at 4,096 tokens.
  Random data (zeros draw less power). Each trial about 2 s, median of three trials.
  `nvidia-smi` samples clocks and power every 0.5 s.
- `pod.py`: creates the pod, runs `run.sh` over SSH, copies `results/` back, terminates the pod.
- Image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, nvcc 12.4.131.

| Card | Driver | Power limit |
|---|---|---|
| RTX 4090 | 595.91.07 | 450 W |
| RTX 3090 | 570.211.01 | 350 W |

## RTX 4090

The card sits at its 450 W limit in every mode, so the raw ratio mixes the cap with the power
limit. Dividing by the SM clock separates them.

| Shape | FP16 acc16, TFLOPS @ MHz | FP16 acc32, TFLOPS @ MHz | Raw ratio | Per-clock ratio |
|---|---|---|---|---|
| sq4096 | 237.4 @ 1965 | 160.7 @ 2520 | 0.68 | 0.53 |
| sq8192 | 224.1 @ 1860 | 148.9 @ 2415 | 0.66 | 0.51 |
| sq16384 | 267.3 @ 2190 | 158.3 @ 2430 | 0.59 | 0.53 |
| qkv | 228.9 @ 1890 | 158.1 @ 2505 | 0.69 | 0.52 |
| gate_up | 235.3 @ 1995 | 163.1 @ 2520 | 0.69 | 0.55 |
| down | 263.9 @ 2220 | 156.4 @ 2460 | 0.59 | 0.54 |

- BF16 acc32: 159.6 to 161.1 TFLOPS on all six shapes. A flat ceiling that does not move with
  shape is a rate cap, not a power limit.
- TF32: 85.0 to 86.2 TFLOPS. INT8: 514.7 to 587.6 TOPS, no cap visible.
- In capped modes the card clocks about 25% higher at the same 450 W: less work per cycle, less
  current, so boost raises the clock back to the limit.

## RTX 3090

| Shape | FP16 acc16 | FP16 acc32 | BF16 acc32 | Raw ratio (FP16) |
|---|---|---|---|---|
| sq4096 | 106.8 | 57.7 | 57.8 | 0.54 |
| sq8192 | 109.5 | 62.9 | 63.9 | 0.57 |
| sq16384 | 116.9 | 66.9 | 67.6 | 0.57 |
| qkv | 94.5 | 60.0 | 60.2 | 0.63 |
| gate_up | 89.6 | 60.6 | 63.7 | 0.68 |
| down | 89.4 | 60.2 | 58.9 | 0.67 |

TFLOPS. TF32: 28.4 to 31.6. INT8: 136.6 to 164.2.

- FP16 acc16 on the square shapes runs at the 350 W limit (1,380 to 1,485 MHz). The
  FP32-accumulate phases run mostly below it, about 270 to 290 W: the capped mode cannot even
  use the power budget.
- Per-row clocks are reliable only for the first three rows: later configurations ran back to
  back and their telemetry segments merged. No per-clock column for that reason.

## Same die, professional parts (batch of 2026-10-26, 09:20 UTC)

Same `run.sh` with a 3 s idle gap between configurations, so telemetry joins row by row
(`analyze.py`). Blackwell parts on `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`.
Per-clock ratio = (FP16 acc32 TFLOPS / SM MHz) / (FP16 acc16 TFLOPS / SM MHz), six shapes.

| Die | GeForce, per-clock ratio | Professional, per-clock ratio |
|---|---|---|
| GA102 | RTX 3090: 0.54 to 0.57 (raw, square) | RTX A6000: 0.95 to 1.06 (raw 0.95 to 1.06) |
| AD102 | RTX 4090: 0.51 to 0.55 | RTX 6000 Ada: 0.89 to 1.04 |
| GB202 | RTX 5090: 0.51 to 0.59 | RTX PRO 6000 Blackwell WS: 0.85 to 1.06 |

- **RTX A6000** (driver 595.91.07, 300 W, sits at 296 to 299 W in both modes, 1,380 to 1,485
  MHz): the two modes are equal raw and per clock. No power confounder at all. The cleanest result.
- **RTX 6000 Ada** (300 W limit, at the limit in every mode, 780 to 1,140 MHz): heavily
  power-bound, so raw ratios (0.67 to 0.90) move with clocks; per clock the modes match.
- **RTX 5090** (575 W, at the limit, driver 595.91.07): capped like the 4090. BF16 acc32 183.1 to
  237.0 TFLOPS, against 286.2 to 319.7 for FP16 acc16.
- **RTX PRO 6000 Blackwell Workstation** (driver 580.178.04): the host reports a 390 W power limit
  on a card NVIDIA rates at 600 W, and the card drew about 300 W at 937 to 1,545 MHz. Absolute
  numbers from this pod say nothing about the card's capability; the per-clock ratio is the only
  usable figure, and it is the noisiest (0.85 to 1.06) because clocks moved most.
- L40: no capacity on RunPod at launch. L40S: see below if it completes.

The two groups do not overlap: GeForce 0.51 to 0.59, professional 0.85 to 1.06.

## Verdict

- **Holds** on both cards for the square shapes: raw ratio 0.54 to 0.57 on the 3090; per-clock
  ratio 0.51 to 0.55 on the 4090.
- **The 4090 raw ratio (0.59 to 0.69) sits between the two thresholds.** Power was named in advance
  as the confounder, but the per-clock normalization was chosen after seeing the data. Say so.
- **Prefill shapes on the 3090 (0.63 to 0.68)** are weaker: FP16 acc16 itself fell to about 90
  TFLOPS there, below the power limit, so those shapes are less efficient in cuBLAS on Ampere.

## What it means for the unlock

FP16 acc16, the mode the cap does not touch, reaches 224 to 267 TFLOPS on a stock 4090 at 450 W,
not 330. An unlocked FP32 accumulation hits the same wall. Expected gain on a stock-power 4090:
about +40 to 65% on large GEMMs, not 2x; less on whole workloads, nothing on bandwidth-bound decode.
Consistent with the gpu-unlock-ada README (application GEMM below the issue-rate gain under a
power limit; cuBLAS +12 to 34%).

## Limits

- One pod per card, community hosts, unknown cooling and board. Not repeated.
- INT8 on the 3090 (about 150 TOPS) is not a clean control: `cublasGemmEx` with this layout is
  likely not the fast IMMA path on Ampere. Do not quote it as a capability.
- Spec figures (4090: about 165 TFLOPS acc32, 330 acc16, dense; 3090: about 71 and 142) are from
  memory. Check against NVIDIA's Ada and Ampere whitepapers before publication.
- Cost: two failed attempts (PATH, a macro bug) plus two full runs, under $1 in total.

## Replication, 2026-10-26, 10:33 to 11:05 UTC

Second runs on different RunPod hosts, same `run.sh`:
- RTX 4090: FP16 acc32 145.0 to 158.3 TFLOPS, per clock 0.52 to 0.55 (first run 148.9 to 163.1, 0.51 to 0.55).
- RTX A6000: acc32 108.8 to 120.8, per clock 0.96 to 1.06 (first run 105.3 to 117.8).
- RTX 6000 Ada: acc32 93.3 to 123.1 at 299 to 300 W, 765 to 960 MHz, per clock 0.88 to 1.04 (first run 94.7 to 127.1).

The delivered-math ranking holds on both hosts: the half-rate RTX 4090 at 450 W delivers more 32-bit-sum math than the full-rate RTX 6000 Ada at its 300 W rating. Independent check: Bizon ResNet-50 FP16, one GPU, RTX 6000 Ada 1800, RTX 4090 1720.
- RTX 3090, second run (11:24 UTC, other host, joined telemetry): acc32 68.7 to 74.1 TFLOPS at 339 to 342 W, 1,770 to 1,785 MHz; per clock 0.51 to 0.54. The first run (57.7 to 66.9 at 270 to 290 W) was throttled below the power limit. Owners' stock baselines (Zagitta issues #114, #17, #148): 62.9 to 80.1.
