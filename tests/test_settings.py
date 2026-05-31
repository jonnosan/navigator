"""test_settings.py — config.yaml loads into the pydantic tree (M0)."""

from __future__ import annotations

from navigator.settings import Settings, default_config_path


def test_default_config_loads() -> None:
    s = Settings.from_yaml(default_config_path())
    # The philosophy invariants the config encodes:
    assert s.codec.name in ("dac", "encodec")
    assert s.codec.roundtrip_corr_min > 0.5  # round-trip test needs a real bar
    assert s.bootstrap.target_pairs >= 1
    assert 0.0 <= s.navigate.default_strength <= 1.0
    assert s.reader.confidence_gate.held_out_pairwise_accuracy > 0.5
    # Generator is opt-in. System must work with it off.
    assert isinstance(s.generator.enabled, bool)
    # Codec latents are the preferred feature source.
    assert s.features.source in ("codec_latents", "mert", "clap")
