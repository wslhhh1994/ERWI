# ERWI: Elastic Reflection Waveform Inversion with SeismicMoE

A research implementation of two-dimensional elastic reflection waveform inversion using the acoustic–elastic coupled wave equation (AECE), CuPy GPU computation, MPI shot-level parallelism, and a PyTorch mixture-of-experts network.

This repository provides the inversion driver and neural network implementation. Model files, observed shot records, data generation scripts, and pretrained weights are not included.

## Features

- Alternating ELSRTM and ERWI: update squared-velocity perturbations in the inner loop and background P-wave/S-wave velocities in the outer loop.
- Staggered-grid finite differences, PML boundaries, and boundary wavefield storage/reconstruction.
- Gradient smoothing, clipping, depth weighting, conjugate-gradient directions, and progressive offset selection.
- SeismicMoE: soft-gated FNO, bidirectional GRU, and CNN experts, two-dimensional rotary positional encoding, a residual bottleneck, cross-trace attention, and a frequency-aware decoder.
- Structure-guided smoothing and optional well constraints; the driver defaults to `USE_WELL = False`.

## Repository Structure

```text
ERWI/
├── nnc_aece_rwi.py   # AECE propagation, ELSRTM/ERWI, and MoE training
├── seismic_moe.py    # Network architecture, training helpers, and self-check
├── requirements.txt # General Python dependencies
├── .gitignore
└── README.md
```

## Installation

Python 3.10 or later is recommended. Full inversion requires an NVIDIA GPU, a compatible NVIDIA driver and CUDA environment, CUDA-enabled PyTorch, CuPy, and an MPI runtime such as Open MPI, MPICH, or MS-MPI on Windows. Dependency versions are not pinned; validate compatibility on the target machine.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell: use the following activation command instead
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Ensure that the installed PyTorch build supports your GPU. Install exactly one CuPy distribution matching your CUDA environment; for CUDA 12.x, for example:

```bash
python -m pip install cupy-cuda12x
```

Do not install multiple CuPy distributions together. Installing Python dependencies does not configure the NVIDIA driver, CUDA environment, or MPI runtime. Full inversion has no CPU fallback.

## Input Data

Run the driver from the repository root. All input and output paths are relative to the current working directory.

### Model Files

Provide these headerless NumPy `float32` binary files:

| File | Purpose | Shape |
| --- | --- | --- |
| `overthrust_450_3_175_marine_iso_3.bin` | Reference model and density | `(450, 3, 175)` |
| `overthrust_450_3_175_marine_linear.bin` | Initial model input | `(450, 3, 175)` |

The dimension order is `(x, parameter, z)`, with channels `[Vp, Vs, density]`, stored in NumPy C order. Velocities are used in m/s; density must match the data generation convention. Each file contains 236250 values, or 945000 bytes.

**Initialization:** the code retains shallow initial values, replaces initial velocities below `bound` with reference velocities smoothed using a Gaussian filter with `sigma=10`, and reads density directly from the reference model. This experimental configuration therefore requires a reference model.

### Observed Shot Records

The default input directory is:

```text
ae_record_overthrust_50_90_f8_dx25_Tn3000_P_iso_cut/
├── 0_pwave_cut.bin
├── 1_pwave_cut.bin
├── ...
├── 49_pwave_cut.bin
├── 0_vx_cut.bin       # Optional velocity component
├── 0_vz_cut.bin       # Optional velocity component
└── ...
```

Each component file contains headerless `float32` values reshaped to `(NR, Tn) = (90, 3000)`, or 1080000 bytes per shot per component. If velocity files are missing, the corresponding velocity data term is disabled for that shot.

**Pressure records must be present.** The current implementation substitutes simulated pressure when a pressure file is missing, potentially producing zero residuals. Continued execution does not establish that observations were loaded correctly.

The default geometry has 50 shots, 90 receivers per shot, a shot spacing of 7 grid cells, and a near offset of 10 grid cells. Receivers lie to the left of the source:

```python
sx[shot] = pml + 100 + shot * 7
rx[shot][i] = sx[shot] - 10 - i
```

Source and receiver depths are both `pml + 1`. These coordinates include PML.

## Configuration and Execution

Edit parameters directly in `nnc_aece_rwi.py`; the driver has no command-line argument interface.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `xn`, `zn` | `450`, `175` | Grid dimensions excluding PML |
| `dx`, `dz` | `25`, `25` | Grid spacing in m |
| `dt`, `Tn` | `0.002`, `3000` | Time step in s and sample count |
| `FM` | `8` | Source dominant frequency in Hz |
| `pml` | `30` | PML thickness in grid cells |
| `N_INNER_PER_CYCLE` | `3` | ELSRTM iterations per cycle |
| `N_OUTER_PER_CYCLE` | `5` | ERWI iterations per cycle |
| `USE_WELL` | `False` | Enable well constraints |
| `X_start` | `70` | Starting lateral index for MoE updates |
| `DATA_DIR` | See above | Shot-record directory |
| `save_dir` | `nn_erwi_iso_result_over_pro2_sm/` | Output directory |

The main loop defaults to `range(0, 100)`. Keep model dimensions, acquisition geometry, sampling, and file layouts consistent when changing the experiment.

### GPU Mapping

The driver originally sets `CUDA_VISIBLE_DEVICES = "0,1,3,4,5"` and `processes_per_gpu = 5`. Device assignment is `rank // processes_per_gpu`: each group of five consecutive ranks shares one visible GPU. Rank 0 also performs MoE training.

Adjust these settings for your machine. At least `ceil(number_of_MPI_ranks / processes_per_gpu)` visible GPUs are required; concurrency must fit GPU memory. The visibility assignment currently follows the CuPy import. Set the same value in the launch environment, or move the assignment before all GPU-library imports, to configure visibility before CUDA initialization.

For a single-GPU, single-rank run, first change the assignment inside the driver to `CUDA_VISIBLE_DEVICES = "0"`, then launch:

```bash
# Linux shell
CUDA_VISIBLE_DEVICES=0 mpiexec -n 1 python nnc_aece_rwi.py
```

Windows PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
mpiexec -n 1 python nnc_aece_rwi.py
```

Increase `-n` after validating inputs and single-rank execution, and recheck GPU mapping. Full inversion requires substantial compute and GPU memory.

### Standalone Network Self-Check

This check requires PyTorch but no inversion datasets. It can use CPU if CUDA is unavailable:

```bash
python seismic_moe.py
```

It exercises forward passes with and without well information, output-shape assertions, and training steps. Input shape: `(n_x, 2, 5, n_z)`; output shape: `(n_x, 2, 1, n_z)`. The two channels represent P-wave and S-wave quantities; 5 is the neighboring-trace window size.

## Outputs

Results are written under `nn_erwi_iso_result_over_pro2_sm/`:

- **`models/`:** `vp_<iter>.bin`, `vs_<iter>.bin`, squared-velocity perturbations `vp_p_<iter>.bin` and `vs_p_<iter>.bin`, density `den_<iter>.bin` saved during outer iterations, objective history `obj_f.txt`, and `moe_weights.pth` after MoE training is triggered.
- **`grads/`:** background and perturbation gradient/search-direction files.
- **`residuals/`:** shot pressure residuals named `<shot>_p.bin`; filenames are reused across iterations.

Model and gradient arrays exclude PML and use headerless `float32` storage with shape `(450, 175)`. Perturbation files represent squared-velocity changes, not direct velocity increments. Inner iterations save velocities computed by adding perturbations to squared background velocities and taking square roots; outer iterations save updated background velocities. Later training overwrites the weights file.

```python
import numpy as np

vp = np.fromfile(
    "nn_erwi_iso_result_over_pro2_sm/models/vp_0.bin",
    dtype=np.float32,
).reshape(450, 175)
```

## Scope and License

This is research code with fixed experimental settings. Full inversion depends on external data, suitable GPU/MPI resources, and consistent configuration. No precomputed inversion results or accuracy guarantees are provided.

No open-source license has been specified. Public availability does not by itself grant unrestricted redistribution or commercial-use rights. Contact the repository owner to confirm permission before reuse.
