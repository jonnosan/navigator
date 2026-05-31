"""Settings — pydantic models mirroring config.yaml.

One YAML on disk, one Settings tree in memory. No hardcoded paths anywhere else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class PathsCfg(BaseModel):
    data_dir: Path
    audio_dir: Path
    code_cache_dir: Path
    db_path: Path


class DeviceCfg(BaseModel):
    preference: Literal["auto", "mps", "cuda", "cpu"] = "auto"


class CodecCfg(BaseModel):
    name: Literal["dac", "encodec"]
    model: str
    sample_rate: int
    roundtrip_corr_min: float = 0.95


class FeaturesCfg(BaseModel):
    source: Literal["codec_latents", "mert", "clap"] = "codec_latents"
    fallback_model: str | None = None


class GeneratorCfg(BaseModel):
    enabled: bool = False
    model: str = "facebook/musicgen-small"


class HeadCfg(BaseModel):
    hidden_dims: list[int]
    dropout: float = 0.0
    type: str | None = None  # boundary head: "pointer" | "crf"


class ReaderTrainingCfg(BaseModel):
    batch_size: int = 16
    max_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 5


class ReaderGateCfg(BaseModel):
    held_out_pairwise_accuracy: float = 0.65
    min_pairs_required: int = 30


class ReaderCfg(BaseModel):
    value_head: HeadCfg
    boundary_head: HeadCfg
    training: ReaderTrainingCfg
    confidence_gate: ReaderGateCfg


class BootstrapCfg(BaseModel):
    target_pairs: int = 40
    diversity_count: int = 200
    pair_selection: Literal["diverse", "random"] = "diverse"


class ReaderGradientCfg(BaseModel):
    enabled: bool = True
    step_size: float = 0.15
    neighbors_k: int = 16


class NavigateCfg(BaseModel):
    default_strength: float = 0.30
    default_count: int = 4
    candidate_multiplier: int = 8
    reader_gradient: ReaderGradientCfg = Field(default_factory=ReaderGradientCfg)


class CorpusModelCfg(BaseModel):
    enabled: bool = False
    kind: Literal["gmm", "vae", "ar"] = "gmm"
    components: int = 32


class ApiCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    reverse_proxy_note: str = ""


class Settings(BaseSettings):
    """Top-level settings tree. Load with `Settings.from_yaml(path)`."""

    paths: PathsCfg
    device: DeviceCfg
    codec: CodecCfg
    features: FeaturesCfg
    generator: GeneratorCfg
    reader: ReaderCfg
    bootstrap: BootstrapCfg
    navigate: NavigateCfg
    corpus_model: CorpusModelCfg
    api: ApiCfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


def default_config_path() -> Path:
    """Repo-root config.yaml — useful as a script/CLI default."""
    return Path(__file__).resolve().parents[2] / "config.yaml"
