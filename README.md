# Cross-Domain Hyperspectral Classification with Semantic Tokens, MoE and an HMA Teacher

Implementation of the architecture described in [instruction.md](instruction.md):
unsupervised cross-scene hyperspectral classification with a 3D/2D CNN semantic
tokenizer, per-domain Transformer encoders whose FFN is replaced by a Top-2
Mixture-of-Experts, a bidirectional coupled Transformer, and representation
distillation against Hull-Moving-Average teachers.

```
X ∈ R^{13×13×13}
   → Conv3D(2³) → Conv3D(2³) → Conv2D(2²)        (semantic tokenizer)
   → 5 semantic tokens + positional encoding
   → Transformer × depth, FFN = Top-2 MoE        (independent per domain)
   → H_s, H_t
        ├── mean pool → h_s → classifier         → L_cls (source labels only)
        ├── coupled cross-attention (both ways)  → C_{s←t}, C_{t←s}
        └── HMA teachers (stop-grad)             → L_dist = Σ 1 − cos(h^T, h^student)
   L_total = L_cls + λ_dist(e)·L_dist + λ_bal·L_balance
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate                # Windows;  source .venv/bin/activate on Linux
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements.txt
```

**Older GPUs.** A wheel only carries kernels for the architectures it was built
for. CUDA 13 wheels start at sm_75 (Turing), so Pascal cards — GTX 10-series,
sm_61 — and Volta need a cu126 or cu118 build:

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.get_arch_list())"   # want an sm_6x entry
```

Without it the run dies inside the first Conv3d with `FIND was unable to find an
engine to execute this computation`. `utils/device.py` checks the compute
capability at startup and reports this instead of letting it surface there.

## Check the installation

```bash
python -m tests.test_phases   # Phases 1-10 of the implementation plan
python smoke_test.py          # end-to-end run on synthetic scenes, CPU, ~1 minute
```

`smoke_test.py` prints the shape at every stage, trains the complete model for a
couple of epochs, evaluates the target domain and round-trips a checkpoint.

## Train

Put the scenes in `data/raw/` (see [data/README.md](data/README.md)), then:

```bash
python tools/inspect_data.py --config configs/houston.yaml   # verify the files load
python train.py --config configs/houston.yaml
python test.py  --checkpoint runs/houston/best.pt
```

Any config value can be overridden on the command line:

```bash
python train.py --config configs/houston.yaml \
    --override train.epochs=200 model.embed_dim=128 model.moe.num_experts=8
```

Each run writes to `output_dir`: `config.yaml` (the resolved config), `train.log`,
`metrics.csv` (per-epoch), `summary.json`, and `last.pt` / `best.pt`.

## Repository layout

| path | contents |
|---|---|
| `data/preprocessing.py` | `.mat` loading, normalisation, PCA spectral reduction, patching |
| `data/registry.py` | file-name and `.mat`-key table for the known benchmarks |
| `data/source_dataset.py`, `data/target_dataset.py` | labelled source / unlabelled target patches |
| `models/semantic_tokenizer.py` | Conv3D → Conv3D → Conv2D + learned attention token pooling |
| `models/moe.py` | experts, Top-k router, load-balancing loss, usage statistics |
| `models/transformer_block.py`, `transformer_encoder.py` | `LN(H + MHSA)` → `LN(H' + MoE)` blocks |
| `models/coupled_attention.py`, `coupled_transformer.py` | bidirectional cross-domain attention |
| `models/hma_teacher.py` | the HMA / EMA / frozen teacher |
| `models/cross_domain_model.py` | assembly, ablation switches, inference options A and B |
| `losses/` | classification, bidirectional distillation, MoE balancing |
| `training/trainer.py`, `training/scheduler.py` | training loop, distillation warm-up, LR schedules |
| `evaluation/` | OA / AA / Kappa / F1, confusion matrix, classification maps |
| `tools/run_ablation.py` | runs the ablation table and collects the results |

## Ablations

`configs/ablation/` holds one config per row of the study in section 28 of the
specification, plus the routing, expert-count, token-count and teacher-type
studies:

```bash
python tools/run_ablation.py --dataset houston --seeds 0 1 2
python tools/run_ablation.py --dataset houston --seeds 0 --include-extra
```

| config | tokenizer | MoE | coupled | distillation | teacher |
|---|---|---|---|---|---|
| `source_only.yaml` | CNN (`depth: 0`) | – | – | – | – |
| `transformer.yaml` | CNN | – | – | – | – |
| `moe.yaml` | CNN | Top-2 | – | – | – |
| `coupled.yaml` | CNN | Top-2 | yes | – | – |
| `distillation.yaml` | CNN | Top-2 | yes | yes | unsmoothed (`copy`) |
| `full.yaml` | CNN | Top-2 | yes | yes | HMA |
| `ema_teacher.yaml` | CNN | Top-2 | yes | yes | EMA |

Plus `frozen_teacher`, `top1_routing`, `experts_2` / `experts_8`, `tokens_1` /
`tokens_10`, `inference_b`, `shared_branches` and `aux_cls_coupled`
(`--include-extra` runs these too).

## Implementation notes and deviations

Points where the specification left a choice open, or where a literal reading does
not train:

* **MoE balancing loss.** The spec defines `p_k` as the *fraction of tokens
  assigned* to expert `k`. Hard assignment counts are piecewise constant in the
  router weights, so that loss has zero gradient and cannot balance anything. The
  default `model.moe.balance_mode: soft` uses the mean router probability instead —
  the usual differentiable surrogate, which is identical in expectation at the
  uniform point. `hard` (literal, gradient-free — for logging) and `switch` (the
  standard sparse-MoE auxiliary loss, mentioned in section 12 as a later
  replacement) are also available.
* **Token pooling attention.** `A = Softmax(F'W_A)` is normalised over the `N`
  spatial positions, so each of the 5 tokens is a convex combination of spatial
  features. `model.tokenizer.softmax_over: token` normalises across tokens instead.
* **HMA window.** The spec gives the WMA cascade but not `n`; the default is
  `window: 16` optimiser steps, with `update_every` to stretch it. Larger windows
  hold more parameter snapshots — `history_device: cpu` moves them off the GPU.
* **What the HMA smooths.** The teachers mirror the *independent* branches
  (tokenizer + positional encoding + encoder), which is what the distillation
  targets `h_s^teacher` / `h_t^teacher` are computed from. Float buffers
  (BatchNorm statistics) are smoothed with the parameters; integer buffers are
  copied.
* **Target classification path — worth attention before the first real run.**
  Inference Option A sends `h_t` through a classifier that only ever saw `h_s`, and
  section 9 of the spec makes the two encoders *fully independent*. Nothing ties the
  two feature spaces to a common coordinate frame: the only pressure on the target
  branch is indirect, through the coupled branch (`H_t` supplies K/V for `C_{s←t}`)
  and the distillation loss.

  This is measurable. On the synthetic scenes, which share class signatures and
  differ only by a per-band gain/offset, two epochs give:

  | branches | source train acc | target OA |
  |---|---|---|
  | independent (spec default) | 91% | 5% (below the 14% chance level) |
  | shared tokenizer + encoder | 91% | 75-80% |

  The source branch learns fine either way, so this is the architecture's structure,
  not a training failure. Real scene pairs are more favourable than this synthetic
  one, but expect the same direction. Three ways to address it, in increasing
  distance from the specification:

  1. `model.aux_cls_on_coupled: true` with `loss.lambda_aux_cls > 0` — cross-entropy
     on the coupled source representation `h_{s←t}`, which pulls the coupled space
     (and through it the target branch) towards the classifier.
  2. `model.share_tokenizer/share_encoder: true` — the "fully shared" and "partially
     shared" variants that section 9 itself lists as later experiments.
  3. `model.inference_mode: B` — classify `h_{t←s}`, which is built from source
     values `V_s` and is therefore already in the classifier's space, at the cost of
     needing source patches at inference time.

  All four combinations have configs in `configs/ablation/`, and the independent /
  Option A default is left exactly as specified.
* **Model selection.** `train.model_selection: target_oa` follows the common
  protocol of this literature but does peek at target labels. Use `last`, or
  `source_val_oa` with `dataset.val_fraction > 0`, for a stricter protocol;
  `summary.json` always reports both the last epoch and the selected checkpoint.
* **Coupled depth > 1.** At each layer both directions read the previous layer's
  output of the other stream, so neither direction sees an already-updated partner.

## Suggested order of work

Following section 27 of the spec — `python -m tests.test_phases` verifies phases
1-10 in one command, and each ablation config isolates one contribution:

1. `tools/inspect_data.py` — confirm the scenes load and the class maps agree.
2. `configs/ablation/source_only.yaml` — CNN baseline, source only.
3. `configs/ablation/transformer.yaml` → `moe.yaml` — is Top-2 MoE better than an FFN? (Q2)
4. `configs/ablation/coupled.yaml` — does bidirectional coupling help? (Q4)
5. `configs/ablation/distillation.yaml` → `full.yaml` → `ema_teacher.yaml` — does
   HMA smoothing beat no smoothing and an EMA? (Q6)
