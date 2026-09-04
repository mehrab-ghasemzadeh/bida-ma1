# Data

Scene files are located **by name**, case-insensitively, by searching the config's
`dataset.root` and then the project root and `data/raw/` (skipping `.venv`, `.git`,
`runs`, …). So any of these work without further edits:

```
Houston/                     data/raw/Houston/          data/raw/
  Houston13.mat                Houston13.mat              Houston13.mat
  Houston13_7gt.mat            Houston13_7gt.mat          Houston13_7gt.mat
  Houston18.mat                Houston18.mat              Houston18.mat
  Houston18_7gt.mat            Houston18_7gt.mat          Houston18_7gt.mat
```

`configs/houston.yaml` uses `root: Houston` — the folder at the project root.
**Houston13 is the source domain, Houston18 the target.**

Check what was found before training:

```bash
python tools/inspect_data.py --root Houston
python tools/inspect_data.py --config configs/houston.yaml
```

## Expected format

Each scene is a `.mat` file (v7 or v7.3), `.npy` or `.npz` holding

* a **cube** `(H, W, B)` — height, width, spectral bands;
* a **ground-truth map** `(H, W)` — `0` = unlabelled, `1..C` = class index.

Source and target scenes must use the **same class indices** and, unless
`spectral_reduction` is set to `none`, the **same number of bands**.

## Benchmarks the registry knows about

| config name | source → target | bands | classes | expected files |
|---|---|---|---|---|
| `houston` | Houston 2013 → Houston 2018 | 48 | 7 | `Houston13.mat`, `Houston13_7gt.mat`, `Houston18.mat`, `Houston18_7gt.mat` |
| `pavia` | Pavia University → Pavia Centre | 102 | 7 | `paviaU.mat`, `paviaU_7gt.mat`, `pavia.mat`, `pavia_7gt.mat` |
| `shanghai_hangzhou` | Shanghai → Hangzhou | 198 | 3 | `DataCube.mat` (keys `DataCube1`, `gt1`, `DataCube2`, `gt2`) |
| `hyrank` | HyRANK Dioni → Loukia | 176 | 12 | `Dioni.mat`, `Dioni_gt_out68.mat`, `Loukia.mat`, `Loukia_gt_out68.mat` |

The `.mat` variable names of the usual public releases (`ori_data` / `map`) are
in `data/registry.py`; add candidates there if your download differs.

## Anything else

Point at the files directly from the config:

```yaml
dataset:
  name: custom
  source_image: data/raw/my_source.mat
  source_gt:    data/raw/my_source_gt.mat
  target_image: data/raw/my_target.mat
  target_gt:    data/raw/my_target_gt.mat
  keys: {source_image: data, source_gt: gt, target_image: data, target_gt: gt}
```

Leave `keys` out to auto-detect (the largest 3D array is the cube, the largest
2D array is the map).

## What the pipeline does to the data

1. **Normalisation** (`dataset.normalisation`, default `standard`) — per-band zero
   mean / unit variance *within each scene*. This removes the per-scene gain and
   offset that would otherwise dominate the domain shift.
2. **Spectral reduction** (`dataset.spectral_reduction`, default `pca`) — the model
   consumes exactly `model.num_bands` = 13 channels. A single PCA is fitted on the
   pixels of *both* scenes so the two domains share one projection. Alternatives:
   `select` (uniformly spaced bands), `head`, `average`, `none`.
3. **Patching** — each cube is reflect-padded by `patch_size // 2` and every
   labelled pixel becomes the centre of a `13 × 13` patch, giving tensors of shape
   `[1, 13, 13, 13]` = `[C, Spectral, H, W]`.

Source labels are the only labels used for training. `dataset.target_pixels`
controls the target pool: `labelled` (default) uses the pixels that carry a
ground-truth label — *without their labels* — which is the transductive protocol
of the cross-scene literature; `all` uses every pixel of the target scene.

`data/raw/` is git-ignored.
