from __future__ import annotations

from pathlib import Path

from src.config.assets import asset_aliases, normalize_asset_name


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
RUNS_DIR = LOGS_DIR / "runs"
EVALUATION_DIR = LOGS_DIR / "evaluation"
DOCS_DIR = BASE_DIR / "docs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_data_path(asset: str) -> Path:
    return RAW_DIR / f"{normalize_asset_name(asset)}_1h.csv"


def resolve_existing_raw_data_path(asset: str) -> Path:
    for alias in asset_aliases(asset):
        candidate = RAW_DIR / f"{alias.lower()}_1h.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No raw data file found for asset '{asset}'.")


def processed_artifact_path(asset: str, suffix: str) -> Path:
    return PROCESSED_DIR / f"{normalize_asset_name(asset)}_{suffix}"


def resolve_processed_artifact_path(asset: str, suffix: str) -> Path:
    for alias in asset_aliases(asset):
        candidate = PROCESSED_DIR / f"{alias.lower()}_{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No processed artifact '{suffix}' found for asset '{asset}'."
    )


def best_checkpoint_path(asset: str) -> Path:
    return MODELS_DIR / f"{normalize_asset_name(asset)}_best.pt"


def final_checkpoint_path(asset: str) -> Path:
    return MODELS_DIR / f"{normalize_asset_name(asset)}_final.pt"


def legacy_checkpoint_candidates(asset: str) -> list[Path]:
    candidates: list[Path] = []
    for alias in asset_aliases(asset):
        candidates.append(MODELS_DIR / f"{alias.lower()}_model.pth")
        candidates.append(MODELS_DIR / f"{alias.lower()}_model_final.pth")
    return candidates


def resolve_checkpoint_path(asset: str, checkpoint: str | None = None) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if path.exists():
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    candidates = [
        best_checkpoint_path(asset),
        final_checkpoint_path(asset),
        *legacy_checkpoint_candidates(asset),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found for asset '{normalize_asset_name(asset)}' in {MODELS_DIR}."
    )
