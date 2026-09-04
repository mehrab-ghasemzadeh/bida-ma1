"""Known cross-scene hyperspectral benchmarks and how to find their files.

Each entry lists candidate file names for the source/target cubes and ground-truth
maps, plus the .mat variable names used by the usual public releases. File lookup
is case-insensitive and tries every candidate, so slightly different downloads
still resolve; anything unusual can be pointed at explicitly from the config with

    dataset:
      name: custom
      source_image: data/raw/my_source.mat
      source_gt:    data/raw/my_source_gt.mat
      ...
      keys: {source_image: ori_data, source_gt: map, ...}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DATASETS: Dict[str, Dict] = {
    "houston": {
        "description": "Houston 2013 -> Houston 2018, 48 shared bands, 7 shared classes.",
        "source_image": ["Houston13.mat", "Houston13_ori.mat", "houston13.mat"],
        "source_gt": ["Houston13_7gt.mat", "Houston13_gt.mat", "houston13_7gt.mat"],
        "target_image": ["Houston18.mat", "Houston18_ori.mat", "houston18.mat"],
        "target_gt": ["Houston18_7gt.mat", "Houston18_gt.mat", "houston18_7gt.mat"],
        "keys": {
            "source_image": "ori_data", "source_gt": "map",
            "target_image": "ori_data", "target_gt": "map",
        },
        "num_classes": 7,
        "class_names": [
            "Grass healthy", "Grass stressed", "Trees", "Water",
            "Residential buildings", "Non-residential buildings", "Road",
        ],
    },
    "pavia": {
        "description": "Pavia University -> Pavia Centre, 102 shared bands, 7 shared classes.",
        "source_image": ["paviaU.mat", "PaviaU.mat"],
        "source_gt": ["paviaU_7gt.mat", "paviaU_gt.mat", "PaviaU_gt.mat"],
        "target_image": ["pavia.mat", "Pavia.mat", "paviaC.mat"],
        "target_gt": ["pavia_7gt.mat", "pavia_gt.mat", "Pavia_gt.mat"],
        "keys": {
            "source_image": "ori_data", "source_gt": "map",
            "target_image": "ori_data", "target_gt": "map",
        },
        "num_classes": 7,
        "class_names": [
            "Tree", "Asphalt", "Brick", "Bitumen", "Shadow", "Meadow", "Bare soil",
        ],
    },
    "shanghai_hangzhou": {
        "description": "Shanghai -> Hangzhou, 198 bands, 3 classes (one shared DataCube.mat).",
        "source_image": ["DataCube.mat", "Shanghai.mat"],
        "source_gt": ["DataCube.mat", "Shanghai_gt.mat"],
        "target_image": ["DataCube.mat", "Hangzhou.mat"],
        "target_gt": ["DataCube.mat", "Hangzhou_gt.mat"],
        "keys": {
            "source_image": "DataCube1", "source_gt": "gt1",
            "target_image": "DataCube2", "target_gt": "gt2",
        },
        "num_classes": 3,
        "class_names": ["Water", "Land/Building", "Plant"],
    },
    "hyrank": {
        "description": "HyRANK Dioni -> Loukia, 176 bands, 12 classes.",
        "source_image": ["Dioni.mat"],
        "source_gt": ["Dioni_gt_out68.mat", "Dioni_gt.mat"],
        "target_image": ["Loukia.mat"],
        "target_gt": ["Loukia_gt_out68.mat", "Loukia_gt.mat"],
        "keys": {
            "source_image": "ori_data", "source_gt": "map",
            "target_image": "ori_data", "target_gt": "map",
        },
        "num_classes": 12,
        "class_names": [
            "Dense urban fabric", "Mineral extraction sites", "Non irrigated arable land",
            "Fruit trees", "Olive groves", "Coniferous forest",
            "Dense sclerophyllous vegetation", "Sparse sclerophyllous vegetation",
            "Sparsely vegetated areas", "Rocks and sand", "Water", "Coastal water",
        ],
    },
}


# Directories that are never worth walking when looking for scene files.
_SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "runs", "checkpoints",
    "logs", "node_modules", ".idea", ".vscode", "site-packages",
}


def _find_file(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    """Search `root` recursively for the first matching candidate (case-insensitive)."""
    wanted = {c.lower() for c in candidates}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS
                             and not d.startswith("."))
        for filename in sorted(filenames):
            if filename.lower() in wanted:
                return Path(dirpath) / filename
    return None


def candidate_roots(root: str | Path) -> List[Path]:
    """Where to look for scene files, in order.

    The configured root comes first; the project root and `data/raw` follow, so a
    scene folder dropped next to the code (e.g. `Houston/`) is found without
    editing the config.
    """
    here = Path(__file__).resolve().parents[1]
    ordered = [Path(root), here / Path(root), here, here / "data" / "raw", Path.cwd()]
    seen, out = set(), []
    for path in ordered:
        resolved = path.resolve()
        if resolved not in seen and path.exists():
            seen.add(resolved)
            out.append(path)
    return out


def resolve_dataset(cfg: Dict, data_root: str | Path = "data/raw") -> Dict:
    """Turn a dataset config into concrete file paths, keys and class metadata."""
    name = str(cfg.get("name", "custom")).lower()
    root = Path(cfg.get("root", data_root))
    roots = candidate_roots(root)

    explicit = {
        field: cfg.get(field)
        for field in ("source_image", "source_gt", "target_image", "target_gt")
    }

    if name == "custom" or name not in DATASETS:
        missing = [field for field, value in explicit.items() if not value]
        if missing:
            raise ValueError(
                f"Unknown dataset '{name}'. Either use one of {sorted(DATASETS)} or "
                f"give explicit paths for: {', '.join(missing)}"
            )
        resolved = {field: Path(value) for field, value in explicit.items()}
        keys = dict(cfg.get("keys", {}) or {})
        class_names = list(cfg.get("class_names", []) or [])
        num_classes = cfg.get("num_classes")
    else:
        spec = DATASETS[name]
        resolved = {}
        problems: List[str] = []
        for field in ("source_image", "source_gt", "target_image", "target_gt"):
            if explicit[field]:
                resolved[field] = Path(explicit[field])
                continue
            if not roots:
                problems.append(f"{field}: no searchable data root (tried '{root}')")
                continue
            found = next(
                (hit for hit in (_find_file(r, spec[field]) for r in roots) if hit), None
            )
            if found is None:
                problems.append(f"{field}: none of {spec[field]} found")
            else:
                resolved[field] = found
        if problems:
            searched = ", ".join(f"'{r}'" for r in roots) or f"'{root}'"
            raise FileNotFoundError(
                f"Could not locate the '{name}' dataset files (searched {searched}):\n  "
                + "\n  ".join(problems)
                + "\nSee data/README.md for the expected layout."
            )
        keys = dict(spec.get("keys", {}))
        keys.update(cfg.get("keys", {}) or {})
        class_names = list(cfg.get("class_names") or spec.get("class_names", []))
        num_classes = cfg.get("num_classes", spec.get("num_classes"))

    for field, path in resolved.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{field} not found: {path}")

    return {
        "name": name,
        "paths": {field: str(path) for field, path in resolved.items()},
        "keys": keys,
        "class_names": class_names,
        "num_classes": num_classes,
    }
