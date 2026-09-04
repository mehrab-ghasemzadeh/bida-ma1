"""Hyperspectral scene loading, normalisation, spectral reduction and patching.

The pipeline turns a pair of scenes into patch datasets of shape [1, S, H, W] with
S = `num_bands` spectral channels and H = W = `patch_size`, as required by the
3D convolutions of the semantic tokenizer.

Steps
-----
1. Load the source and target cubes (H, W, B) and their ground-truth maps (H, W),
   where 0 marks unlabelled pixels and 1..C the classes.
2. Normalise each scene (per-band standardisation by default, which removes the
   per-scene illumination/gain offset that otherwise dominates the domain shift).
3. Reduce the spectral dimension to `num_bands` - by default with a PCA fitted
   jointly on both scenes, so the two domains share one projection.
4. Reflect-pad each cube by patch_size // 2 and index patches by centre pixel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------- loading
def _candidate_arrays(mat: Dict) -> Dict[str, np.ndarray]:
    return {
        key: value
        for key, value in mat.items()
        if not key.startswith("__") and isinstance(value, np.ndarray)
    }


def load_mat(path: str | Path, key: Optional[str] = None, ndim: Optional[int] = None
             ) -> np.ndarray:
    """Load an array from a .mat / .npy / .npz file.

    Handles both classic (scipy) and v7.3 / HDF5 (h5py) .mat files. When `key` is
    None, the largest array with the requested number of dimensions is used.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path))
    if suffix == ".npz":
        bundle = np.load(path)
        if key is not None:
            return np.asarray(bundle[key])
        arrays = {k: np.asarray(bundle[k]) for k in bundle.files}
        return _pick_array(arrays, ndim, path)

    try:
        from scipy.io import loadmat

        mat = loadmat(str(path))
        arrays = _candidate_arrays(mat)
    except (NotImplementedError, ValueError):
        import h5py

        with h5py.File(path, "r") as handle:
            arrays = {
                k: np.array(handle[k]).T  # HDF5 .mat files are stored transposed
                for k in handle.keys()
                if isinstance(handle[k], h5py.Dataset)
            }

    if key is not None:
        if key not in arrays:
            raise KeyError(
                f"Key '{key}' not in {path.name}. Available keys: {sorted(arrays)}"
            )
        return np.asarray(arrays[key])
    return _pick_array(arrays, ndim, path)


def _pick_array(arrays: Dict[str, np.ndarray], ndim: Optional[int],
                path: Path) -> np.ndarray:
    candidates = [v for v in arrays.values() if ndim is None or v.ndim == ndim]
    if not candidates:
        raise ValueError(
            f"No {ndim}D array found in {path.name}; available: "
            + ", ".join(f"{k}{v.shape}" for k, v in arrays.items())
        )
    return np.asarray(max(candidates, key=lambda a: a.size))


# --------------------------------------------------------------- normalisation
def normalise_cube(cube: np.ndarray, method: str = "standard") -> np.ndarray:
    """Normalise an (H, W, B) cube.

    ``standard``  - per-band zero mean / unit variance within the scene.
    ``minmax``    - per-band scaling to [0, 1] within the scene.
    ``global``    - single min/max over the whole cube.
    ``none``      - unchanged.
    """
    cube = cube.astype(np.float32)
    if method == "none":
        return cube
    if method == "standard":
        mean = cube.mean(axis=(0, 1), keepdims=True)
        std = cube.std(axis=(0, 1), keepdims=True)
        return (cube - mean) / (std + 1e-6)
    if method == "minmax":
        lo = cube.min(axis=(0, 1), keepdims=True)
        hi = cube.max(axis=(0, 1), keepdims=True)
        return (cube - lo) / (hi - lo + 1e-6)
    if method == "global":
        lo, hi = float(cube.min()), float(cube.max())
        return (cube - lo) / (hi - lo + 1e-6)
    raise ValueError(f"Unknown normalisation method: {method}")


# ---------------------------------------------------------- spectral reduction
def reduce_spectral(
    source: np.ndarray,
    target: np.ndarray,
    num_bands: int,
    method: str = "pca",
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bring both scenes to exactly `num_bands` spectral channels.

    ``pca``     - one PCA fitted on the pixels of both scenes, so source and target
                  share the same projection (the default; requires equal band counts).
    ``select``  - uniformly spaced band subset.
    ``head``    - the first `num_bands` bands.
    ``average`` - contiguous band groups averaged into `num_bands` channels.
    ``none``    - unchanged; both scenes must already have `num_bands` bands.
    """
    if source.shape[-1] != target.shape[-1] and method in {"pca", "select", "head", "average"}:
        raise ValueError(
            f"Source has {source.shape[-1]} bands and target {target.shape[-1]}. "
            "Trim the scenes to a common band set before spectral reduction."
        )

    bands = source.shape[-1]
    if method == "none" or bands == num_bands:
        if source.shape[-1] != num_bands:
            raise ValueError(
                f"Spectral reduction is 'none' but the scenes have {bands} bands "
                f"while the model expects {num_bands}"
            )
        return source, target

    if num_bands > bands:
        raise ValueError(f"Cannot expand {bands} bands to {num_bands}")

    if method == "select":
        idx = np.linspace(0, bands - 1, num_bands).round().astype(int)
        return source[..., idx], target[..., idx]

    if method == "head":
        return source[..., :num_bands], target[..., :num_bands]

    if method == "average":
        edges = np.linspace(0, bands, num_bands + 1).round().astype(int)
        src = np.stack([source[..., edges[i]:edges[i + 1]].mean(-1) for i in range(num_bands)], -1)
        tgt = np.stack([target[..., edges[i]:edges[i + 1]].mean(-1) for i in range(num_bands)], -1)
        return src.astype(np.float32), tgt.astype(np.float32)

    if method == "pca":
        from sklearn.decomposition import PCA

        src_flat = source.reshape(-1, bands)
        tgt_flat = target.reshape(-1, bands)
        joint = np.concatenate([src_flat, tgt_flat], axis=0)
        # Fitting on a subsample keeps large scenes cheap without changing the
        # components appreciably.
        max_fit = 200_000
        if joint.shape[0] > max_fit:
            rng = np.random.default_rng(seed)
            joint = joint[rng.choice(joint.shape[0], max_fit, replace=False)]
        pca = PCA(n_components=num_bands, random_state=seed)
        pca.fit(joint)
        src = pca.transform(src_flat).reshape(*source.shape[:2], num_bands)
        tgt = pca.transform(tgt_flat).reshape(*target.shape[:2], num_bands)
        return src.astype(np.float32), tgt.astype(np.float32)

    raise ValueError(f"Unknown spectral reduction method: {method}")


# --------------------------------------------------------------------- patches
def pad_cube(cube: np.ndarray, patch_size: int, mode: str = "reflect") -> np.ndarray:
    """Pad the spatial dimensions so every pixel can be the centre of a patch."""
    half = patch_size // 2
    return np.pad(cube, ((half, half), (half, half), (0, 0)), mode=mode)


def labelled_coordinates(gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Coordinates of labelled pixels and their 0-based labels."""
    rows, cols = np.nonzero(gt)
    coords = np.stack([rows, cols], axis=1)
    labels = gt[rows, cols].astype(np.int64) - 1
    return coords, labels


def all_coordinates(gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Every pixel of the scene, with label -1 where the ground truth is 0."""
    rows, cols = np.meshgrid(np.arange(gt.shape[0]), np.arange(gt.shape[1]), indexing="ij")
    coords = np.stack([rows.ravel(), cols.ravel()], axis=1)
    labels = gt.ravel().astype(np.int64) - 1
    return coords, labels


def sample_per_class(
    coords: np.ndarray,
    labels: np.ndarray,
    num_per_class: int,
    seed: int = 0,
    allow_fewer: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw at most `num_per_class` samples from each class without replacement."""
    rng = np.random.default_rng(seed)
    keep: List[np.ndarray] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        if len(idx) > num_per_class:
            idx = rng.choice(idx, num_per_class, replace=False)
        elif not allow_fewer:
            raise ValueError(f"Class {cls} has only {len(idx)} samples")
        keep.append(idx)
    selected = np.sort(np.concatenate(keep))
    return coords[selected], labels[selected]


class HSIPatchDataset(Dataset):
    """Patches cut from a padded cube, indexed by their centre pixel.

    Yields `(patch, label)` with patch of shape [1, S, patch, patch] - the
    [B, 1, Spectral, Height, Width] layout the 3D convolutions expect - and label
    -1 for unlabelled pixels.
    """

    def __init__(
        self,
        cube: np.ndarray,
        coords: np.ndarray,
        labels: np.ndarray,
        patch_size: int = 13,
        pad_mode: str = "reflect",
        already_padded: bool = False,
        augment: bool = False,
        seed: int = 0,
    ):
        self.patch_size = patch_size
        self.padded = cube if already_padded else pad_cube(cube, patch_size, pad_mode)
        self.coords = np.asarray(coords, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.coords)

    def patch_at(self, row: int, col: int) -> np.ndarray:
        size = self.patch_size
        window = self.padded[row:row + size, col:col + size, :]      # (H, W, S)
        return np.ascontiguousarray(window.transpose(2, 0, 1))       # (S, H, W)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = self.coords[index]
        patch = self.patch_at(int(row), int(col))
        if self.augment:
            # The global numpy RNG is used deliberately: `utils.seed.worker_init_fn`
            # reseeds it per dataloader worker, so workers do not draw identical
            # augmentations.
            k = int(np.random.randint(4))
            if k:
                patch = np.rot90(patch, k, axes=(1, 2))
            if np.random.random() < 0.5:
                patch = patch[:, :, ::-1]
            patch = np.ascontiguousarray(patch)
        tensor = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0)  # [1, S, H, W]
        return tensor, torch.tensor(int(self.labels[index]))

    def class_counts(self, num_classes: int) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        valid = self.labels[self.labels >= 0]
        for cls, count in zip(*np.unique(valid, return_counts=True)):
            counts[int(cls)] = count
        return counts


@dataclass
class ScenePair:
    """A prepared source/target scene pair, ready to be wrapped in datasets."""

    source_cube: np.ndarray
    source_gt: np.ndarray
    target_cube: np.ndarray
    target_gt: np.ndarray
    num_classes: int
    class_names: Sequence[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"source {self.source_cube.shape} gt {self.source_gt.shape} | "
            f"target {self.target_cube.shape} gt {self.target_gt.shape} | "
            f"{self.num_classes} classes"
        )


def prepare_scene_pair(
    source_image: str | Path,
    source_gt: str | Path,
    target_image: str | Path,
    target_gt: str | Path,
    num_bands: int = 13,
    normalisation: str = "standard",
    spectral_reduction: str = "pca",
    keys: Optional[Dict[str, Optional[str]]] = None,
    class_names: Sequence[str] = (),
    seed: int = 0,
) -> ScenePair:
    """Load, normalise and spectrally reduce a source/target scene pair."""
    keys = keys or {}
    src = load_mat(source_image, keys.get("source_image"), ndim=3).astype(np.float32)
    src_gt = load_mat(source_gt, keys.get("source_gt"), ndim=2).astype(np.int64)
    tgt = load_mat(target_image, keys.get("target_image"), ndim=3).astype(np.float32)
    tgt_gt = load_mat(target_gt, keys.get("target_gt"), ndim=2).astype(np.int64)

    if src.shape[:2] != src_gt.shape:
        raise ValueError(f"Source cube {src.shape} does not match its map {src_gt.shape}")
    if tgt.shape[:2] != tgt_gt.shape:
        raise ValueError(f"Target cube {tgt.shape} does not match its map {tgt_gt.shape}")

    src = normalise_cube(src, normalisation)
    tgt = normalise_cube(tgt, normalisation)
    src, tgt = reduce_spectral(src, tgt, num_bands, spectral_reduction, seed=seed)

    num_classes = int(max(src_gt.max(), tgt_gt.max()))
    return ScenePair(src, src_gt, tgt, tgt_gt, num_classes, list(class_names))
