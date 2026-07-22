# Fleet: Cooperative Scheduling for Megakernels on Multi-Die GPUs

Fleet is a cooperative task scheduling system for LLM inference on AMD MI350 GPUs. It extends the [Mirage Persistent Kernel (MPK)](https://arxiv.org/abs/2512.22219) framework with XCD-aware chiplet scheduling, enabling workers on the same execution core die (XCD) to cooperate on shared tasks for better L2 cache reuse and reduced memory bandwidth pressure.

**Paper**: [Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs](https://arxiv.org/abs/2604.15379) (submitted to ASPLOS 2027)

## Key Ideas

Modern multi-die GPUs like the AMD MI350 have 8 XCDs, each with its own 4 MB L2 cache. Traditional megakernel schedulers dispatch tasks to workers without regard for die topology, causing L2 thrashing when multiple XCDs read the same weights simultaneously.

Fleet solves this with **chiplet scheduling**: all ~30 workers on an XCD process tiles of the same GEMM together, keeping weights in the local L2 cache. Subsequent decode iterations reuse cached weights instead of re-reading from HBM. Fleet also introduces **hierarchical synchronization** that scopes signaling to L2-local counters within each XCD, reducing cross-chiplet fence traffic by 14.5x.

### Results (Qwen3-8B, bf16, AMD MI350)

| Batch Size | Mirage (ms) | Fleet M-tile (ms) | Fleet M-split (ms) | Speedup |
|:----------:|:-----------:|:------------------:|:------------------:|:-------:|
| 1          | 7.993       | 7.076              | 7.012              | 1.13x   |
| 32         | 15.186      | 12.357             | 12.923             | 1.23x   |
| 64         | 23.343      | 18.176             | 21.923             | 1.28x   |

Fleet achieves up to **1.28x** speedup over Mirage MPK on dense models.

## Hardware Requirements

- AMD Instinct MI350 (gfx950)
  - 248 CUs organized into 8 XCDs
  - 4 MB L2 cache per XCD, 256 MB MALL
  - HBM3 bandwidth: 5.3 TB/s
- ROCm 7.0+

## Installation

```bash
# Clone the repository
git clone git@github.com:AMD-RAD/fleet.git
cd fleet
git checkout amd_mi350

# Build from source
pip install -e . -v
export MIRAGE_HOME=$(pwd)
```

### Prerequisites

- Python 3.8+
- PyTorch 2.4+ (ROCm build)
- ROCm 7.0+ with hipcc compiler
- CMake 3.24+

### Download the Model

```bash
# Qwen3-8B (required for Figure 6)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B')"
```

## Reproducing Figure 6: Decode Latency

Figure 6 compares decode-only time per output token (TPOT) across batch sizes for four configurations on Qwen3-8B (bf16).

### Configurations

| Mode | Description | Environment Variables |
|------|-------------|----------------------|
| `mirage_mpk` | Baseline persistent kernel (no cooperative scheduling) | `USE_CK_FMHA=1 USE_FUSED_SILU=1` |
| `fleet` | Fleet with M-tile cooperative scheduling | `USE_GANG=1 USE_CK_FMHA=1 USE_FUSED_SILU=1 GANG_M_TILES_GATEUP=1` |
| `fleet_msplit` | Fleet with M-split (disjoint M-tile assignment) | `USE_GANG=1 USE_GANG_M_SPLIT=1 USE_CK_FMHA=1 USE_FUSED_SILU=1` |

### Step 1: Verify Setup

Make sure `MIRAGE_HOME` is set and the model is downloaded:

```bash
export MIRAGE_HOME=$(pwd)
python3 -c "import mirage; print('Mirage OK')"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B')"
```

### Step 2: Run a Single Configuration

To run one mode at a specific batch size (e.g., Fleet M-tile at BS=1):

```bash
# Delete cached kernel to force recompilation with new flags
rm -rf permanent_output_dir

# Run Fleet M-tile at batch size 1
HIP_VISIBLE_DEVICES=0 USE_GANG=1 USE_CK_FMHA=1 USE_FUSED_SILU=1 GANG_M_TILES_GATEUP=1 \
  python3 demo/qwen3/demo.py --use-mirage \
  --model Qwen/Qwen3-8B \
  --max-num-batched-tokens 1 --max-num-batched-requests 1 \
  --max-seq-length 1200 --max-new-tokens 1024 \
  --ignore-eos \
  --prompt "Explain the key differences between RISC and CISC processor architectures, including their instruction set design philosophies, pipeline implementations, memory access patterns, register file organization, microcode usage, and how modern processors have evolved to blur the traditional boundaries between these two approaches in terms of performance, power efficiency, and overall transistor utilization."
```

The output will include a line like:
```
Decode: 1141 tokens in 1141 iterations ≈ 7783.2ms total (6.820ms/token)
```

The **decode-only TPOT** (ms/token) is the metric reported in Figure 6.

**Important**: You must delete `permanent_output_dir/` before switching between modes, as the megakernel is compiled with mode-specific flags.

To sweep all modes and batch sizes, repeat Step 2 for each configuration in the table above, deleting `permanent_output_dir/` between runs. Each mode x batch size takes 3-10 minutes (including kernel compilation).

### Expected Results

Reproduced results (AMD MI350, single GPU):

| BS | Mirage (ms) | Fleet M-tile (ms) | Fleet M-split (ms) | Speedup (M-tile) |
|:--:|:-----------:|:------------------:|:------------------:|:----------------:|
| 1  | 7.993       | 7.076              | 7.012              | 1.13x            |
| 2  | 8.505       | 7.597              | 7.486              | 1.12x            |
| 4  | 9.100       | 8.064              | 7.972              | 1.13x            |
| 8  | 9.833       | 8.587              | 8.554              | 1.15x            |
| 16 | 10.824      | 9.357              | 9.317              | 1.16x            |
| 32 | 15.186      | 12.357             | 12.923             | 1.23x            |
| 64 | 23.343      | 18.176             | 21.923             | 1.28x            |

Paper reference values: Mirage 7.83ms, Fleet M-tile 6.82ms, Fleet M-split 6.73ms at BS=1.
Reproduced numbers may vary by ~5-15% depending on GPU thermal state, HBM bandwidth, and silicon revision.

## How Fleet Works

Fleet builds on Mirage MPK's persistent megakernel architecture. The key additions:

1. **Chiplet scheduling**: Workers on the same XCD cooperatively process tiles of the same GEMM. This keeps weight data in the XCD's 4 MB L2 cache instead of re-reading from HBM on every iteration.

2. **Hierarchical synchronization**: Fleet matches signaling to memory scopes to minimize cross-chiplet coherence traffic. Within a chiplet task, workers increment XCD-local L2 counters without fences. Only the last worker on each XCD issues a single `buffer_wbl2` fence and updates the global HBM counter. Schedulers poll the global counter to dispatch downstream tasks. This reduces total signals from 25,200 (naive per-tile) to 1,830 (per-event per-XCD) — a **14.5x reduction** — and confines most coordination to the local L2 cache, avoiding expensive device-scope atomics across the inter-chiplet fabric.

3. **M-tile windowed traversal**: Each XCD walks through M-tiles of the output matrix in a windowed pattern, maximizing temporal L2 reuse of weight columns.

4. **Chiplet SiLU fusion**: The SiLU activation and element-wise multiply are fused into the gate_up projection task, eliminating an intermediate buffer write.

## Citation

```bibtex
@inproceedings{chowdhary2027fleet,
  title={Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs},
  author={Sangeeta Chowdhary and Xinhao Cheng and Zhihao Zhang and Yu Zhou and Jianan Ji and Jinchen Jiang and Zepeng Zhao and Ziruo Xiao and Zihao Ye and Yingyi Huang and Ruihang Lai and Hongyi Jin and Bohan Hou and Mengdi Wu and Yixin Dong and Anthony Yip and Zihao Ye and Songting Wang and Wenqin Yang and Xupeng Miao and Tianqi Chen and Zhihao Jia},
  year={2027},
  booktitle={Proceedings of the 32nd ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)},
}
```

## License

Apache License 2.0.
